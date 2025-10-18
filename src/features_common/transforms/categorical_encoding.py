"""
Categorical Feature Encoding for Unified Data-Driven Pipeline.

This module provides comprehensive categorical feature encoding capabilities
including one-hot encoding, ordinal encoding, and target encoding with VectorBT integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, LabelEncoder,
    TargetEncoder, LabelBinarizer
)
from sklearn.model_selection import KFold
import logging

# Import VectorBT components
try:
    from src.features_common.vectorbt_extensions.unified_manager import UnifiedVectorizationManager
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

class CategoricalEncoder:
    """
    Comprehensive categorical feature encoder for the unified pipeline.

    Supports multiple encoding strategies with automatic detection of
    categorical features and appropriate encoding selection. Integrates
    with VectorBT for optimized performance when available.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the categorical encoder."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize VectorBT components if available
        self.vectorbt_manager = None
        if VECTORBT_AVAILABLE:
            try:
                self.vectorbt_manager = UnifiedVectorizationManager()
                tprint_success("✅ VectorBT components initialized for categorical encoding")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT initialization failed: {e}")

        # Encoding strategies configuration
        self.encoding_strategies = {
            'one_hot': OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
            'label': LabelEncoder(),
            'target': TargetEncoder(smooth='auto', cv=5),
            'binary': LabelBinarizer()
        }

        # Feature detection configuration
        self.categorical_threshold = self.config.get('categorical_threshold', 0.05)  # 5% unique values
        self.max_categories = self.config.get('max_categories', 50)
        self.min_frequency = self.config.get('min_frequency', 10)
        self.use_vectorbt = self.config.get('use_vectorbt', VECTORBT_AVAILABLE)

        # Encoding results tracking
        self.encoding_results = {}
        self.feature_mappings = {}

        tprint_success("✅ CategoricalEncoder initialized")

    def detect_categorical_features(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically detect categorical features in the dataset.

        Args:
            data: Input DataFrame

        Returns:
            Dictionary mapping column names to detected categorical types
        """
        categorical_features = {}

        for col in data.columns:
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(data[col]):
                continue

            # Check for object/string columns
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                unique_ratio = data[col].nunique() / len(data)

                if unique_ratio <= self.categorical_threshold:
                    # Low cardinality - likely categorical
                    if data[col].nunique() <= self.max_categories:
                        categorical_features[col] = 'low_cardinality'
                    else:
                        categorical_features[col] = 'high_cardinality'
                else:
                    # High cardinality - might be ordinal or need special handling
                    categorical_features[col] = 'high_cardinality'

        tprint_info(f"📊 Detected {len(categorical_features)} categorical features")
        return categorical_features

    def select_encoding_strategy(self, feature_name: str, feature_type: str,
                               data: pd.Series, target: Optional[pd.Series] = None) -> str:
        """
        Select appropriate encoding strategy for a categorical feature.

        Args:
            feature_name: Name of the feature
            feature_type: Detected type of the feature
            data: Feature data
            target: Target variable (for target encoding)

        Returns:
            Selected encoding strategy name
        """
        unique_count = data.nunique()
        missing_ratio = data.isnull().sum() / len(data)

        # Strategy selection logic
        if unique_count <= 2:
            return 'binary'
        elif unique_count <= 10 and missing_ratio < 0.1:
            return 'one_hot'
        elif unique_count <= 20 and data.dtype == 'object':
            # Check if values seem ordinal
            if self._is_ordinal_feature(data):
                return 'ordinal'
            else:
                return 'one_hot'
        elif unique_count > 20 and target is not None:
            return 'target'
        else:
            return 'ordinal'

    def _is_ordinal_feature(self, data: pd.Series) -> bool:
        """Check if a feature appears to be ordinal."""
        try:
            # Try to convert to numeric to see if it's ordinal
            numeric_data = pd.to_numeric(data.dropna(), errors='coerce')
            return not numeric_data.isnull().any()
        except:
            return False

    def encode_features(self, data: pd.DataFrame,
                       target: Optional[pd.Series] = None,
                       feature_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical features in the dataset.

        Args:
            data: Input DataFrame
            target: Target variable for target encoding
            feature_list: Specific features to encode (if None, auto-detect)

        Returns:
            DataFrame with encoded categorical features
        """
        tprint_info("🔤 Starting categorical feature encoding")

        # Detect categorical features if not provided
        if feature_list is None:
            categorical_features = self.detect_categorical_features(data)
            feature_list = list(categorical_features.keys())

        if not feature_list:
            tprint_info("ℹ️ No categorical features found")
            return data

        # Try VectorBT optimization for large datasets
        if (self.use_vectorbt and VECTORBT_AVAILABLE and
            len(data) > 1000 and len(feature_list) > 1):
            try:
                tprint_info("🚀 Using VectorBT for optimized categorical encoding")
                encoded_data = self._vectorbt_encode_features(data, feature_list, target)
                if encoded_data is not None:
                    tprint_success("✅ VectorBT categorical encoding completed")
                    return encoded_data
                else:
                    tprint_warning("⚠️ VectorBT encoding failed, falling back to traditional methods")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT encoding error: {e}, using traditional methods")

        # Traditional encoding approach
        encoded_data = data.copy()

        for feature in feature_list:
            if feature not in data.columns:
                tprint_warning(f"⚠️ Feature {feature} not found in data")
                continue

            try:
                # Select encoding strategy
                feature_type = 'low_cardinality'  # Default
                if feature in categorical_features:
                    feature_type = categorical_features[feature]

                strategy = self.select_encoding_strategy(
                    feature, feature_type, data[feature], target
                )

                # Apply encoding
                encoded_feature = self._apply_encoding(
                    data[feature], strategy, feature, target
                )

                if encoded_feature is not None:
                    # Replace original feature with encoded version
                    if isinstance(encoded_feature, pd.DataFrame):
                        # Multiple columns (e.g., one-hot encoding)
                        encoded_data = encoded_data.drop(columns=[feature])
                        encoded_data = pd.concat([encoded_data, encoded_feature], axis=1)
                        tprint_success(f"✅ Encoded {feature} using {strategy}: {len(encoded_feature.columns)} columns")
                    else:
                        # Single column (e.g., ordinal encoding)
                        encoded_data[feature] = encoded_feature
                        tprint_success(f"✅ Encoded {feature} using {strategy}")

            except Exception as e:
                tprint_error(f"❌ Error encoding feature {feature}: {e}")
                continue

        tprint_success(f"✅ Categorical encoding completed: {len(feature_list)} features processed")
        return encoded_data

    def _vectorbt_encode_features(self, data: pd.DataFrame,
                                 feature_list: List[str],
                                 target: Optional[pd.Series] = None) -> Optional[pd.DataFrame]:
        """Use VectorBT for optimized categorical encoding."""
        try:
            if not self.vectorbt_manager:
                return None

            # Prepare categorical data
            categorical_data = data[feature_list].copy()

            # Use VectorBT for one-hot encoding (most common case)
            encoded_data = data.copy()

            for feature in feature_list:
                if categorical_data[feature].nunique() <= 10:  # Suitable for one-hot
                    # Use VectorBT's optimized one-hot encoding
                    one_hot_result = self.vectorbt_manager.one_hot_encode(
                        categorical_data[feature],
                        feature_name=feature
                    )

                    if one_hot_result is not None:
                        # Drop original feature and add encoded columns
                        encoded_data = encoded_data.drop(columns=[feature])
                        encoded_data = pd.concat([encoded_data, one_hot_result], axis=1)

                        # Store mapping
                        self.feature_mappings[feature] = {
                            'strategy': 'one_hot_vectorbt',
                            'vectorbt': True,
                            'columns': list(one_hot_result.columns)
                        }
                else:
                    # Use ordinal encoding for high cardinality
                    ordinal_result = self.vectorbt_manager.ordinal_encode(
                        categorical_data[feature],
                        feature_name=feature
                    )

                    if ordinal_result is not None:
                        encoded_data[feature] = ordinal_result

                        # Store mapping
                        self.feature_mappings[feature] = {
                            'strategy': 'ordinal_vectorbt',
                            'vectorbt': True
                        }

            return encoded_data

        except Exception as e:
            tprint_error(f"❌ VectorBT categorical encoding failed: {e}")
            return None

    def _apply_encoding(self, data: pd.Series, strategy: str,
                       feature_name: str, target: Optional[pd.Series] = None) -> Union[pd.Series, pd.DataFrame, None]:
        """Apply specific encoding strategy to a feature."""
        try:
            if strategy == 'one_hot':
                return self._apply_one_hot_encoding(data, feature_name)
            elif strategy == 'ordinal':
                return self._apply_ordinal_encoding(data, feature_name)
            elif strategy == 'binary':
                return self._apply_binary_encoding(data, feature_name)
            elif strategy == 'target' and target is not None:
                return self._apply_target_encoding(data, target, feature_name)
            else:
                tprint_warning(f"⚠️ Unknown encoding strategy: {strategy}")
                return None

        except Exception as e:
            tprint_error(f"❌ Error applying {strategy} encoding to {feature_name}: {e}")
            return None

    def _apply_one_hot_encoding(self, data: pd.Series, feature_name: str) -> pd.DataFrame:
        """Apply one-hot encoding to a feature."""
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Handle missing values
        data_clean = data.fillna('missing')

        # Fit and transform
        encoded = encoder.fit_transform(data_clean.values.reshape(-1, 1))

        # Create column names
        categories = encoder.categories_[0]
        column_names = [f"{feature_name}_{cat}" for cat in categories]

        # Create DataFrame
        encoded_df = pd.DataFrame(encoded, columns=column_names, index=data.index)

        # Store mapping for later use
        self.feature_mappings[feature_name] = {
            'strategy': 'one_hot',
            'encoder': encoder,
            'categories': categories,
            'vectorbt': False
        }

        return encoded_df

    def _apply_ordinal_encoding(self, data: pd.Series, feature_name: str) -> pd.Series:
        """Apply ordinal encoding to a feature."""
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Handle missing values
        data_clean = data.fillna('missing')

        # Fit and transform
        encoded = encoder.fit_transform(data_clean.values.reshape(-1, 1)).flatten()

        # Create Series
        encoded_series = pd.Series(encoded, index=data.index, name=feature_name)

        # Store mapping for later use
        self.feature_mappings[feature_name] = {
            'strategy': 'ordinal',
            'encoder': encoder,
            'categories': encoder.categories_[0],
            'vectorbt': False
        }

        return encoded_series

    def _apply_binary_encoding(self, data: pd.Series, feature_name: str) -> pd.Series:
        """Apply binary encoding to a feature."""
        encoder = LabelBinarizer()

        # Handle missing values
        data_clean = data.fillna('missing')

        # Fit and transform
        encoded = encoder.fit_transform(data_clean)

        # Create Series
        encoded_series = pd.Series(encoded, index=data.index, name=feature_name)

        # Store mapping for later use
        self.feature_mappings[feature_name] = {
            'strategy': 'binary',
            'encoder': encoder,
            'classes': encoder.classes_,
            'vectorbt': False
        }

        return encoded_series

    def _apply_target_encoding(self, data: pd.Series, target: pd.Series,
                              feature_name: str) -> pd.Series:
        """Apply target encoding to a feature."""
        encoder = TargetEncoder(smooth='auto', cv=5)

        # Handle missing values
        data_clean = data.fillna('missing')

        # Fit and transform
        encoded = encoder.fit_transform(data_clean.values.reshape(-1, 1), target.values)

        # Create Series
        encoded_series = pd.Series(encoded.flatten(), index=data.index, name=feature_name)

        # Store mapping for later use
        self.feature_mappings[feature_name] = {
            'strategy': 'target',
            'encoder': encoder,
            'vectorbt': False
        }

        return encoded_series

    def get_encoding_summary(self) -> Dict[str, Any]:
        """Get summary of encoding operations."""
        summary = {
            'total_features_encoded': len(self.feature_mappings),
            'encoding_strategies_used': {},
            'feature_details': {},
            'vectorbt_usage': 0,
            'traditional_usage': 0
        }

        for feature, mapping in self.feature_mappings.items():
            strategy = mapping['strategy']
            is_vectorbt = mapping.get('vectorbt', False)

            summary['encoding_strategies_used'][strategy] = summary['encoding_strategies_used'].get(strategy, 0) + 1

            if is_vectorbt:
                summary['vectorbt_usage'] += 1
            else:
                summary['traditional_usage'] += 1

            summary['feature_details'][feature] = {
                'strategy': strategy,
                'categories_count': len(mapping.get('categories', mapping.get('classes', []))),
                'vectorbt': is_vectorbt
            }

        return summary

    def inverse_transform(self, data: pd.DataFrame, feature_name: str) -> pd.Series:
        """Inverse transform an encoded feature back to original values."""
        if feature_name not in self.feature_mappings:
            tprint_warning(f"⚠️ No encoding mapping found for feature {feature_name}")
            return data[feature_name]

        mapping = self.feature_mappings[feature_name]
        strategy = mapping['strategy']
        is_vectorbt = mapping.get('vectorbt', False)

        try:
            if is_vectorbt:
                tprint_warning("⚠️ Inverse transform for VectorBT encoding not implemented")
                return data[feature_name]
            elif strategy == 'one_hot':
                # For one-hot encoding, find the original column
                original_categories = mapping['categories']
                # This is complex for one-hot, would need to reconstruct
                tprint_warning("⚠️ Inverse transform for one-hot encoding not implemented")
                return data[feature_name]
            else:
                # For ordinal, binary, and target encoding
                encoder = mapping['encoder']
                encoded_values = data[feature_name].values.reshape(-1, 1)
                inverse_encoded = encoder.inverse_transform(encoded_values)
                return pd.Series(inverse_encoded.flatten(), index=data.index, name=feature_name)

        except Exception as e:
            tprint_error(f"❌ Error in inverse transform for {feature_name}: {e}")
            return data[feature_name]
