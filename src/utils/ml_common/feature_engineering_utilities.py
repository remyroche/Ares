"""
Feature Engineering Utilities

This module provides comprehensive feature engineering utilities extracted from training steps
to eliminate code duplication and provide consistent feature engineering across all steps.

Key Features:
- Feature metadata generation and categorization
- Feature validation and quality checks
- Common feature creation patterns
- Feature interaction and selection utilities
- Integration with ML Common utilities
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import numpy as np

# Import ML Common utilities
from src.utils.ml_common import (
    DataQualityUtilities,
    FeatureSelectionFramework,
    MLTrainingSafeguards
)

# Import step06 utilities for feature engineering
from src.utils.step06_utilities import (
    EnhancedFeatureEngineering,
    Step06UtilityContainer,
    get_utility_container
)

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class FeatureEngineeringUtilities:
    """
    Feature engineering utilities for all training steps.
    
    This provides common feature engineering patterns and utilities
    extracted from multiple training step implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature engineering utilities."""
        self.config = config or {}
        self.logger = logger.getChild('FeatureEngineeringUtilities')
        
        # Initialize ML Common utilities
        self.data_quality = DataQualityUtilities()
        self.safeguards = MLTrainingSafeguards()
        
        # Initialize utility container for dependency injection
        self.utility_container = get_utility_container(config)
        
        # Initialize enhanced feature engineering
        self.feature_engine = EnhancedFeatureEngineering(config)
        
        # Standard feature engineering settings
        self.standard_settings = {
            'enable_technical_indicators': True,
            'enable_statistical_features': True,
            'enable_lag_features': True,
            'enable_interaction_features': True,
            'enable_regime_features': True,
            'enable_wavelet_features': True,
            'enable_multi_timeframe_features': True,
            'max_lags': 10,
            'max_interactions': 50,
            'feature_interaction_degree': 2,
            'timeframes': ['5m', '15m', '1h', '4h'],
            'chunk_size': 100000,
            'max_features': 500
        }
        
        # Update with user configuration
        feature_config = self.config.get('feature_engineering_config', {})
        self.standard_settings.update(feature_config)
        
        self.logger.info("🚀 Feature Engineering Utilities initialized")
    
    def generate_feature_metadata(self, features: pd.DataFrame, feature_type: str) -> Dict[str, Any]:
        """
        Generate metadata about created features.
        
        Args:
            features: DataFrame containing features
            feature_type: Type of features created
            
        Returns:
            Feature metadata dictionary
        """
        try:
            metadata = {
                'feature_type': feature_type,
                'total_features': len(features.columns),
                'feature_names': list(features.columns),
                'data_shape': features.shape,
                'feature_categories': self._categorize_features(features),
                'created_at': datetime.now().isoformat(),
                'settings_used': self.standard_settings
            }
            
            # Add feature statistics
            if len(features) > 0:
                metadata['feature_statistics'] = {
                    'mean_values': features.mean().to_dict(),
                    'std_values': features.std().to_dict(),
                    'min_values': features.min().to_dict(),
                    'max_values': features.max().to_dict(),
                    'missing_values': features.isnull().sum().to_dict()
                }
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error generating feature metadata: {e}")
            return {'error': str(e)}
    
    def _categorize_features(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize features by type.
        
        Args:
            features: DataFrame containing features
            
        Returns:
            Dictionary mapping feature categories to feature names
        """
        try:
            categories = {
                'technical_indicators': [],
                'statistical_features': [],
                'lag_features': [],
                'interaction_features': [],
                'regime_features': [],
                'wavelet_features': [],
                'multi_timeframe_features': [],
                'other': []
            }
            
            for col in features.columns:
                col_lower = col.lower()
                
                if any(indicator in col_lower for indicator in ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic']):
                    categories['technical_indicators'].append(col)
                elif any(stat in col_lower for stat in ['mean', 'std', 'var', 'skew', 'kurt']):
                    categories['statistical_features'].append(col)
                elif 'lag' in col_lower or '_t-' in col_lower:
                    categories['lag_features'].append(col)
                elif 'interaction' in col_lower or '_x_' in col_lower:
                    categories['interaction_features'].append(col)
                elif 'regime' in col_lower:
                    categories['regime_features'].append(col)
                elif 'wavelet' in col_lower:
                    categories['wavelet_features'].append(col)
                elif any(tf in col_lower for tf in ['5m', '15m', '1h', '4h']):
                    categories['multi_timeframe_features'].append(col)
                else:
                    categories['other'].append(col)
            
            # Remove empty categories
            categories = {k: v for k, v in categories.items() if v}
            
            return categories
            
        except Exception as e:
            self.logger.warning(f"Error categorizing features: {e}")
            return {'error': str(e)}
    
    def validate_features(self, features: pd.DataFrame, data_type: str = 'features') -> Dict[str, Any]:
        """
        Validate feature quality and consistency.
        
        Args:
            features: DataFrame containing features
            data_type: Type of data for validation
            
        Returns:
            Validation result dictionary
        """
        try:
            self.logger.info(f"🔍 Validating {data_type} features...")
            
            # Use data quality utilities for validation
            validation_result = self.data_quality.analyze_data_quality(features)
            
            # Add feature-specific validations
            feature_validation = {
                'feature_count': len(features.columns),
                'feature_names_valid': all(isinstance(col, str) for col in features.columns),
                'numeric_features': len(features.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(features.select_dtypes(include=['object', 'category']).columns),
                'constant_features': self._find_constant_features(features),
                'high_correlation_pairs': self._find_high_correlation_pairs(features)
            }
            
            validation_result['feature_validation'] = feature_validation
            
            # Add warnings for feature-specific issues
            if feature_validation['constant_features']:
                validation_result['warnings'].append(f"Found {len(feature_validation['constant_features'])} constant features")
            
            if feature_validation['high_correlation_pairs']:
                validation_result['warnings'].append(f"Found {len(feature_validation['high_correlation_pairs'])} high correlation pairs")
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"Error validating features: {e}")
            return {
                'passed': False,
                'errors': [f"Feature validation error: {e}"],
                'warnings': []
            }
    
    def _find_constant_features(self, features: pd.DataFrame) -> List[str]:
        """Find features with constant values."""
        try:
            constant_features = []
            for col in features.columns:
                if features[col].nunique() <= 1:
                    constant_features.append(col)
            return constant_features
        except Exception as e:
            self.logger.warning(f"Error finding constant features: {e}")
            return []
    
    def _find_high_correlation_pairs(self, features: pd.DataFrame, threshold: float = 0.95) -> List[Dict[str, Any]]:
        """Find highly correlated feature pairs."""
        try:
            numeric_features = features.select_dtypes(include=[np.number])
            if len(numeric_features.columns) < 2:
                return []
            
            correlation_matrix = numeric_features.corr()
            high_corr_pairs = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = abs(correlation_matrix.iloc[i, j])
                    if corr_value > threshold:
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            return high_corr_pairs
        except Exception as e:
            self.logger.warning(f"Error finding high correlation pairs: {e}")
            return []
    
    def create_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic features (technical indicators only).
        
        Args:
            data: Input OHLCV data
            
        Returns:
            DataFrame with basic features
        """
        try:
            self.logger.info("Creating basic features...")
            
            # Use enhanced feature engineering for basic features
            features = self.feature_engine.create_technical_indicators(
                data=data,
                enable_sma=True,
                enable_ema=True,
                enable_rsi=True,
                enable_macd=True,
                enable_bollinger_bands=True,
                enable_stochastic=True
            )
            
            return features
            
        except Exception as e:
            self.logger.exception(f"Error creating basic features: {e}")
            raise
    
    def create_standard_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create standard features (technical indicators + statistical features).
        
        Args:
            data: Input OHLCV data
            
        Returns:
            DataFrame with standard features
        """
        try:
            self.logger.info("Creating standard features...")
            
            # Start with basic features
            features = self.create_basic_features(data)
            
            # Add statistical features
            if self.standard_settings.get('enable_statistical_features', True):
                statistical_features = self.feature_engine.create_statistical_features(
                    data=data,
                    enable_rolling_stats=True,
                    enable_volatility_features=True,
                    enable_momentum_features=True
                )
                features = pd.concat([features, statistical_features], axis=1)
            
            # Add lag features
            if self.standard_settings.get('enable_lag_features', True):
                lag_features = self.feature_engine.create_lag_features(
                    data=data,
                    max_lags=self.standard_settings.get('max_lags', 10)
                )
                features = pd.concat([features, lag_features], axis=1)
            
            return features
            
        except Exception as e:
            self.logger.exception(f"Error creating standard features: {e}")
            raise
    
    def create_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features (all feature types).
        
        Args:
            data: Input OHLCV data
            
        Returns:
            DataFrame with comprehensive features
        """
        try:
            self.logger.info("Creating comprehensive features...")
            
            # Start with standard features
            features = self.create_standard_features(data)
            
            # Add interaction features
            if self.standard_settings.get('enable_interaction_features', True):
                interaction_features = self.feature_engine.create_interaction_features(
                    data=features,
                    max_interactions=self.standard_settings.get('max_interactions', 50),
                    interaction_degree=self.standard_settings.get('feature_interaction_degree', 2)
                )
                features = pd.concat([features, interaction_features], axis=1)
            
            # Add regime features
            if self.standard_settings.get('enable_regime_features', True):
                regime_features = self.feature_engine.create_regime_features(
                    data=data,
                    enable_regime_indicators=True,
                    enable_regime_transitions=True
                )
                features = pd.concat([features, regime_features], axis=1)
            
            # Add wavelet features
            if self.standard_settings.get('enable_wavelet_features', True):
                wavelet_features = self.feature_engine.create_wavelet_features(
                    data=data,
                    enable_wavelet_decomposition=True,
                    enable_wavelet_energy=True
                )
                features = pd.concat([features, wavelet_features], axis=1)
            
            # Add multi-timeframe features
            if self.standard_settings.get('enable_multi_timeframe_features', True):
                multi_timeframe_features = self.feature_engine.create_multi_timeframe_features(
                    data=data,
                    timeframes=self.standard_settings.get('timeframes', ['5m', '15m', '1h', '4h'])
                )
                features = pd.concat([features, multi_timeframe_features], axis=1)
            
            # Limit features if specified
            max_features = self.standard_settings.get('max_features', 500)
            if len(features.columns) > max_features:
                self.logger.warning(f"Limiting features from {len(features.columns)} to {max_features}")
                # Select top features by variance
                feature_variances = features.var().sort_values(ascending=False)
                top_features = feature_variances.head(max_features).index
                features = features[top_features]
            
            return features
            
        except Exception as e:
            self.logger.exception(f"Error creating comprehensive features: {e}")
            raise
    
    def select_features(self, features: pd.DataFrame, targets: pd.Series, 
                       method: str = 'mrmr', n_features: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Select features using specified method.
        
        Args:
            features: Feature DataFrame
            targets: Target series
            method: Selection method ('mrmr', 'importance', 'rfe', 'correlation', 'mutual_info')
            n_features: Number of features to select (if None, auto-select)
            
        Returns:
            Tuple of (selected_features, selection_info)
        """
        try:
            self.logger.info(f"Selecting features using {method} method...")
            
            # Use FeatureSelectionFramework from ml_common
            feature_selector = FeatureSelectionFramework()
            
            # Perform feature selection
            selection_result = feature_selector.select_features(
                X=features,
                y=targets,
                method=method,
                n_features=n_features
            )
            
            selected_features = features[selection_result['selected_features']]
            selection_info = {
                'method': method,
                'n_features_selected': len(selection_result['selected_features']),
                'selected_features': selection_result['selected_features'],
                'feature_scores': selection_result.get('feature_scores', {}),
                'selection_criteria': selection_result.get('selection_criteria', {})
            }
            
            self.logger.info(f"✅ Selected {len(selection_result['selected_features'])} features using {method}")
            
            return selected_features, selection_info
            
        except Exception as e:
            self.logger.exception(f"Error selecting features: {e}")
            raise
    
    def get_feature_importance_analysis(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """
        Analyze feature importance using multiple methods.
        
        Args:
            features: Feature DataFrame
            targets: Target series
            
        Returns:
            Feature importance analysis results
        """
        try:
            self.logger.info("Analyzing feature importance...")
            
            # Use FeatureSelectionFramework for importance analysis
            feature_selector = FeatureSelectionFramework()
            
            # Get importance scores from multiple methods
            importance_analysis = feature_selector.analyze_feature_importance(
                X=features,
                y=targets,
                methods=['mutual_info', 'f_score', 'chi2']
            )
            
            # Add feature statistics
            importance_analysis['feature_statistics'] = {
                'total_features': len(features.columns),
                'numeric_features': len(features.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(features.select_dtypes(include=['object', 'category']).columns),
                'constant_features': len(self._find_constant_features(features)),
                'high_correlation_pairs': len(self._find_high_correlation_pairs(features))
            }
            
            return importance_analysis
            
        except Exception as e:
            self.logger.exception(f"Error analyzing feature importance: {e}")
            return {'error': str(e)}
    
    def get_feature_engineering_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering capabilities."""
        return {
            'config': self.config,
            'standard_settings': self.standard_settings,
            'feature_engine_info': {
                'engine_type': 'EnhancedFeatureEngineering',
                'available_methods': [
                    'create_technical_indicators',
                    'create_statistical_features',
                    'create_lag_features',
                    'create_interaction_features',
                    'create_regime_features',
                    'create_wavelet_features',
                    'create_multi_timeframe_features'
                ]
            },
            'timestamp': datetime.now().isoformat()
        }


# Global instance for easy access
_global_feature_utilities = None

def get_feature_engineering_utilities(config: Optional[Dict[str, Any]] = None) -> FeatureEngineeringUtilities:
    """Get feature engineering utilities instance."""
    return FeatureEngineeringUtilities(config)


# Convenience functions
def generate_feature_metadata(features: pd.DataFrame, feature_type: str) -> Dict[str, Any]:
    """Generate feature metadata using utilities."""
    utils = get_feature_engineering_utilities()
    return utils.generate_feature_metadata(features, feature_type)


def validate_features(features: pd.DataFrame, data_type: str = 'features') -> Dict[str, Any]:
    """Validate features using utilities."""
    utils = get_feature_engineering_utilities()
    return utils.validate_features(features, data_type)


def create_features(data: pd.DataFrame, feature_type: str = 'comprehensive') -> pd.DataFrame:
    """Create features using utilities."""
    utils = get_feature_engineering_utilities()
    
    if feature_type == 'basic':
        return utils.create_basic_features(data)
    elif feature_type == 'standard':
        return utils.create_standard_features(data)
    elif feature_type == 'comprehensive':
        return utils.create_comprehensive_features(data)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def select_features(features: pd.DataFrame, targets: pd.Series, 
                   method: str = 'mrmr', n_features: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select features using utilities."""
    utils = get_feature_engineering_utilities()
    return utils.select_features(features, targets, method, n_features)


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Test feature engineering utilities
    utils = FeatureEngineeringUtilities()
    
    print("=== Creating Features ===")
    features = utils.create_comprehensive_features(data)
    print(f"Created {len(features.columns)} features")
    
    print("\n=== Feature Metadata ===")
    metadata = utils.generate_feature_metadata(features, 'comprehensive')
    print(f"Feature categories: {list(metadata['feature_categories'].keys())}")
    print(f"Total features: {metadata['total_features']}")
    
    print("\n=== Feature Validation ===")
    validation = utils.validate_features(features)
    print(f"Validation passed: {validation['passed']}")
    print(f"Warnings: {validation.get('warnings', [])}")
    
    print("\n=== Feature Selection ===")
    targets = pd.Series(np.random.randint(0, 2, len(features)), name='target')
    selected_features, selection_info = utils.select_features(features, targets, 'mrmr', 10)
    print(f"Selected {len(selected_features.columns)} features")
    print(f"Selection method: {selection_info['method']}")
    
    print("\n=== Feature Importance Analysis ===")
    importance_analysis = utils.get_feature_importance_analysis(features, targets)
    print(f"Importance methods: {list(importance_analysis.keys())}")
    print(f"Feature statistics: {importance_analysis.get('feature_statistics', {})}")