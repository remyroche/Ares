"""
Unified Feature Engineering Infrastructure

This module provides unified feature engineering across all training steps using
EnhancedFeatureEngineering from step06_utilities, replacing multiple feature 
engineering implementations.

Key Features:
- Unified feature engineering using EnhancedFeatureEngineering
- Consolidates 15+ feature engineering files into 2-3 utility-based steps
- Standardized feature engineering approaches across all steps
- Automatic feature validation and quality checks
- Integration with ML Common utilities
- Comprehensive error handling and logging
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import numpy as np

# Import pipeline infrastructure utilities
from src.utils.ml_common.pipeline_infrastructure import (
    create_simple_step_function,
    create_data_processing_step_function
)

# Import configuration management utilities
from src.utils.ml_common.configuration_management import (
    validate_config,
    validate_and_fix_config
)

# Import data quality utilities
from src.utils.ml_common.data_quality_utilities import (
    validate_data_quality,
    clean_data,
    generate_quality_report
)

# Import step06 utilities for feature engineering
from src.utils.step06_utilities import (
    EnhancedFeatureEngineering,
    Step06UtilityContainer,
    get_utility_container
)

# Import ML Common utilities
from src.utils.ml_common import (
    DataQualityUtilities,
    FeatureSelectionFramework,
    MLTrainingSafeguards
)

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class UnifiedFeatureEngineeringManager:
    """
    Unified feature engineering manager for all training steps.
    
    This replaces multiple feature engineering implementations with a unified
    approach using EnhancedFeatureEngineering from step06_utilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize unified feature engineering manager."""
        self.config = validate_and_fix_config(config, 'feature_engineering')
        self.logger = logger.getChild('UnifiedFeatureEngineeringManager')
        
        # Initialize ML Common utilities
        self.data_quality = DataQualityUtilities()
        self.safeguards = MLTrainingSafeguards()
        
        # Initialize utility container for dependency injection
        self.utility_container = get_utility_container(config)
        
        # Initialize enhanced feature engineering
        self.feature_engine = EnhancedFeatureEngineering(config)
        
        # Feature engineering configuration
        self.feature_config = self.config.get('feature_engineering_config', {})
        
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
        self.standard_settings.update(self.feature_config)
        
        self.logger.info("🚀 Unified Feature Engineering Manager initialized")
    
    async def create_features(self, data: pd.DataFrame, feature_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Create features using unified approach.
        
        Args:
            data: Input data
            feature_type: Type of features to create ('basic', 'standard', 'comprehensive')
            
        Returns:
            Feature engineering result
        """
        try:
            self.logger.info(f"🔧 Creating {feature_type} features...")
            
            # Validate input data
            data_validation = validate_data_quality(data, 'ohlcv', 'comprehensive')
            if not data_validation['passed']:
                self.logger.warning(f"Input data quality issues: {data_validation['errors']}")
            
            # Create features based on type
            if feature_type == 'basic':
                features = await self._create_basic_features(data)
            elif feature_type == 'standard':
                features = await self._create_standard_features(data)
            elif feature_type == 'comprehensive':
                features = await self._create_comprehensive_features(data)
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
            
            # Validate created features
            features_validation = validate_data_quality(features, 'features', 'comprehensive')
            
            # Generate feature metadata
            feature_metadata = self._generate_feature_metadata(features, feature_type)
            
            # Generate quality report
            quality_report = generate_quality_report(features, 'features')
            
            return {
                'features': features,
                'feature_metadata': feature_metadata,
                'features_validation': features_validation,
                'quality_report': quality_report,
                'feature_type': feature_type,
                'data_validation': data_validation
            }
            
        except Exception as e:
            self.logger.exception(f"Error creating features: {e}")
            raise
    
    async def _create_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create basic features (technical indicators only)."""
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
    
    async def _create_standard_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create standard features (technical indicators + statistical features)."""
        try:
            self.logger.info("Creating standard features...")
            
            # Start with basic features
            features = await self._create_basic_features(data)
            
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
    
    async def _create_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features (all feature types)."""
        try:
            self.logger.info("Creating comprehensive features...")
            
            # Start with standard features
            features = await self._create_standard_features(data)
            
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
    
    def _generate_feature_metadata(self, features: pd.DataFrame, feature_type: str) -> Dict[str, Any]:
        """Generate metadata about created features."""
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
        """Categorize features by type."""
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


# Simplified feature engineering step functions
async def unified_feature_engineering_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified feature engineering logic using EnhancedFeatureEngineering.
    
    Args:
        config: Configuration dictionary
        pipeline_state: Current pipeline state
        
    Returns:
        Feature engineering result
    """
    logger.info("🔧 Starting unified feature engineering...")
    
    try:
        # Get data from pipeline state
        data = pipeline_state.get('data')
        if data is None:
            raise ValueError("No data found in pipeline state for feature engineering")
        
        # Initialize unified feature engineering manager
        feature_manager = UnifiedFeatureEngineeringManager(config)
        
        # Determine feature type from configuration
        feature_type = config.get('feature_type', 'comprehensive')
        
        # Create features
        result = await feature_manager.create_features(data, feature_type)
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in unified feature engineering: {e}")
        raise


async def basic_feature_engineering_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Basic feature engineering logic (technical indicators only)."""
    logger.info("🔧 Starting basic feature engineering...")
    
    try:
        # Get data from pipeline state
        data = pipeline_state.get('data')
        if data is None:
            raise ValueError("No data found in pipeline state for feature engineering")
        
        # Initialize unified feature engineering manager
        feature_manager = UnifiedFeatureEngineeringManager(config)
        
        # Create basic features
        result = await feature_manager.create_features(data, 'basic')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in basic feature engineering: {e}")
        raise


async def standard_feature_engineering_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Standard feature engineering logic (technical indicators + statistical features)."""
    logger.info("🔧 Starting standard feature engineering...")
    
    try:
        # Get data from pipeline state
        data = pipeline_state.get('data')
        if data is None:
            raise ValueError("No data found in pipeline state for feature engineering")
        
        # Initialize unified feature engineering manager
        feature_manager = UnifiedFeatureEngineeringManager(config)
        
        # Create standard features
        result = await feature_manager.create_features(data, 'standard')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in standard feature engineering: {e}")
        raise


async def comprehensive_feature_engineering_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive feature engineering logic (all feature types)."""
    logger.info("🔧 Starting comprehensive feature engineering...")
    
    try:
        # Get data from pipeline state
        data = pipeline_state.get('data')
        if data is None:
            raise ValueError("No data found in pipeline state for feature engineering")
        
        # Initialize unified feature engineering manager
        feature_manager = UnifiedFeatureEngineeringManager(config)
        
        # Create comprehensive features
        result = await feature_manager.create_features(data, 'comprehensive')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in comprehensive feature engineering: {e}")
        raise


# Create step functions
unified_feature_engineering = create_data_processing_step_function("unified_feature_engineering", unified_feature_engineering_logic)
basic_feature_engineering = create_data_processing_step_function("basic_feature_engineering", basic_feature_engineering_logic)
standard_feature_engineering = create_data_processing_step_function("standard_feature_engineering", standard_feature_engineering_logic)
comprehensive_feature_engineering = create_data_processing_step_function("comprehensive_feature_engineering", comprehensive_feature_engineering_logic)


class SimplifiedFeatureEngineering:
    """
    Simplified feature engineering using unified infrastructure.
    
    This replaces multiple feature engineering implementations with a unified
    approach using EnhancedFeatureEngineering from step06_utilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simplified feature engineering."""
        self.config = validate_and_fix_config(config, 'feature_engineering')
        self.logger = logger.getChild('SimplifiedFeatureEngineering')
        
        # Initialize unified feature engineering manager
        self.feature_manager = UnifiedFeatureEngineeringManager(self.config)
        
        self.logger.info("🚀 Simplified Feature Engineering initialized")
    
    async def create_features(self, data: pd.DataFrame, feature_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Create features using unified approach.
        
        Args:
            data: Input data
            feature_type: Type of features to create
            
        Returns:
            Feature engineering result
        """
        try:
            self.logger.info(f"🚀 Creating {feature_type} features...")
            
            # Create features
            result = await self.feature_manager.create_features(data, feature_type)
            
            self.logger.info(f"✅ Feature engineering completed: {result['feature_metadata']['total_features']} features created")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Feature engineering error: {e}")
            raise
    
    def get_feature_engineering_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering capabilities."""
        return self.feature_manager.get_feature_engineering_summary()


# Backward compatibility wrappers
class AdvancedFeatureEngineeringStep(SimplifiedFeatureEngineering):
    """Backward compatibility wrapper for AdvancedFeatureEngineeringStep."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("🔄 Using backward compatibility wrapper for AdvancedFeatureEngineeringStep")


class FeatureEngineeringStep(SimplifiedFeatureEngineering):
    """Backward compatibility wrapper for FeatureEngineeringStep."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("🔄 Using backward compatibility wrapper for FeatureEngineeringStep")


# Example usage and testing
async def example_feature_engineering():
    """Example of using the unified feature engineering."""
    
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
    
    # Configuration for different feature types
    configs = [
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'feature_type': 'basic',
            'feature_engineering_config': {
                'enable_technical_indicators': True,
                'enable_statistical_features': False,
                'enable_lag_features': False
            }
        },
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'feature_type': 'standard',
            'feature_engineering_config': {
                'enable_technical_indicators': True,
                'enable_statistical_features': True,
                'enable_lag_features': True,
                'max_lags': 5
            }
        },
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'feature_type': 'comprehensive',
            'feature_engineering_config': {
                'enable_technical_indicators': True,
                'enable_statistical_features': True,
                'enable_lag_features': True,
                'enable_interaction_features': True,
                'enable_regime_features': True,
                'enable_wavelet_features': True,
                'enable_multi_timeframe_features': True,
                'max_lags': 10,
                'max_interactions': 20,
                'max_features': 100
            }
        }
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n=== Testing Feature Engineering Type {i+1}: {config['feature_type']} ===")
        
        # Create simplified feature engineering
        feature_engine = SimplifiedFeatureEngineering(config)
        
        # Create features
        result = await feature_engine.create_features(data, config['feature_type'])
        
        # Get summary
        summary = feature_engine.get_feature_engineering_summary()
        
        print(f"Feature type: {result['feature_type']}")
        print(f"Total features: {result['feature_metadata']['total_features']}")
        print(f"Feature categories: {list(result['feature_metadata']['feature_categories'].keys())}")
        print(f"Data shape: {result['features'].shape}")
        
        results.append((result, summary))
    
    return results


# Main execution
async def main():
    """Main execution function."""
    try:
        results = await example_feature_engineering()
        print("✅ Feature engineering example completed successfully")
        return results
    except Exception as e:
        logger.exception(f"Feature engineering example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())