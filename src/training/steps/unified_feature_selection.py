"""
Unified Feature Selection Infrastructure

This module provides unified feature selection across all training steps using
Step08AdvancedFeatureSelection from step08_utilities, replacing multiple feature 
selection implementations.

Key Features:
- Unified feature selection using Step08AdvancedFeatureSelection
- Consolidates multiple feature selection files into utility-based steps
- Standardized feature selection approaches across all steps
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

# Import new simplified infrastructure
from .simplified_pipeline_infrastructure import (
    create_simple_step_function,
    create_data_processing_step_function
)

# Import standardized validation
from .standardized_config_validation import (
    validate_config,
    validate_and_fix_config
)

# Import unified data quality
from .unified_data_quality import (
    validate_data_quality,
    clean_data,
    generate_quality_report
)

# Import step08 utilities for feature selection
from src.utils.step08_utilities import (
    Step08AdvancedFeatureSelection,
    Step08UtilityContainer,
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


class UnifiedFeatureSelectionManager:
    """
    Unified feature selection manager for all training steps.
    
    This replaces multiple feature selection implementations with a unified
    approach using Step08AdvancedFeatureSelection from step08_utilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize unified feature selection manager."""
        self.config = validate_and_fix_config(config, 'feature_selection')
        self.logger = logger.getChild('UnifiedFeatureSelectionManager')
        
        # Initialize ML Common utilities
        self.data_quality = DataQualityUtilities()
        self.safeguards = MLTrainingSafeguards()
        
        # Initialize utility container for dependency injection
        self.utility_container = get_utility_container(config)
        
        # Initialize advanced feature selection
        self.feature_selector = Step08AdvancedFeatureSelection(config)
        
        # Feature selection configuration
        self.selection_config = self.config.get('feature_selection_config', {})
        
        # Standard feature selection settings
        self.standard_settings = {
            'enable_mrmr_selection': True,
            'enable_importance_selection': True,
            'enable_rfe_selection': True,
            'enable_correlation_selection': True,
            'enable_mutual_info_selection': True,
            'enable_regime_specific_selection': True,
            'enable_stability_analysis': True,
            'enable_feature_interaction_analysis': True,
            'n_features': 50,
            'selection_threshold': 0.01,
            'stability_threshold': 0.6,
            'correlation_threshold': 0.95,
            'mutual_info_threshold': 0.01,
            'max_features': 200,
            'min_features': 10
        }
        
        # Update with user configuration
        self.standard_settings.update(self.selection_config)
        
        self.logger.info("🚀 Unified Feature Selection Manager initialized")
    
    async def select_features(self, features: pd.DataFrame, targets: pd.Series, 
                             selection_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Select features using unified approach.
        
        Args:
            features: Input features
            targets: Target values
            selection_type: Type of selection ('basic', 'standard', 'comprehensive')
            
        Returns:
            Feature selection result
        """
        try:
            self.logger.info(f"🎯 Selecting features using {selection_type} approach...")
            
            # Validate input data
            features_validation = validate_data_quality(features, 'features', 'comprehensive')
            targets_validation = validate_data_quality(targets, 'targets', 'standard')
            
            if not features_validation['passed']:
                self.logger.warning(f"Features validation issues: {features_validation['errors']}")
            
            if not targets_validation['passed']:
                self.logger.warning(f"Targets validation issues: {targets_validation['errors']}")
            
            # Select features based on type
            if selection_type == 'basic':
                selected_features = await self._select_basic_features(features, targets)
            elif selection_type == 'standard':
                selected_features = await self._select_standard_features(features, targets)
            elif selection_type == 'comprehensive':
                selected_features = await self._select_comprehensive_features(features, targets)
            else:
                raise ValueError(f"Unknown selection type: {selection_type}")
            
            # Validate selected features
            selected_validation = validate_data_quality(selected_features, 'features', 'comprehensive')
            
            # Generate selection metadata
            selection_metadata = self._generate_selection_metadata(features, selected_features, selection_type)
            
            # Generate quality report
            quality_report = generate_quality_report(selected_features, 'selected_features')
            
            return {
                'selected_features': selected_features,
                'selection_metadata': selection_metadata,
                'selected_validation': selected_validation,
                'features_validation': features_validation,
                'targets_validation': targets_validation,
                'quality_report': quality_report,
                'selection_type': selection_type
            }
            
        except Exception as e:
            self.logger.exception(f"Error selecting features: {e}")
            raise
    
    async def _select_basic_features(self, features: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """Select features using basic approach (correlation-based only)."""
        try:
            self.logger.info("Selecting features using basic approach...")
            
            # Use correlation-based selection
            selected_features = self.feature_selector.select_features_by_correlation(
                features=features,
                targets=targets,
                threshold=self.standard_settings.get('correlation_threshold', 0.95)
            )
            
            return selected_features
            
        except Exception as e:
            self.logger.exception(f"Error in basic feature selection: {e}")
            raise
    
    async def _select_standard_features(self, features: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """Select features using standard approach (correlation + importance)."""
        try:
            self.logger.info("Selecting features using standard approach...")
            
            # Start with basic selection
            basic_features = await self._select_basic_features(features, targets)
            
            # Add importance-based selection
            if self.standard_settings.get('enable_importance_selection', True):
                importance_features = self.feature_selector.select_features_by_importance(
                    features=features,
                    targets=targets,
                    n_features=self.standard_settings.get('n_features', 50)
                )
                
                # Combine selections
                combined_features = pd.concat([basic_features, importance_features], axis=1)
                # Remove duplicates
                selected_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            else:
                selected_features = basic_features
            
            return selected_features
            
        except Exception as e:
            self.logger.exception(f"Error in standard feature selection: {e}")
            raise
    
    async def _select_comprehensive_features(self, features: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """Select features using comprehensive approach (all methods)."""
        try:
            self.logger.info("Selecting features using comprehensive approach...")
            
            # Start with standard selection
            standard_features = await self._select_standard_features(features, targets)
            
            # Add MRMR selection
            if self.standard_settings.get('enable_mrmr_selection', True):
                mrmr_features = self.feature_selector.select_features_by_mrmr(
                    features=features,
                    targets=targets,
                    n_features=self.standard_settings.get('n_features', 50)
                )
                
                # Combine with standard features
                combined_features = pd.concat([standard_features, mrmr_features], axis=1)
                # Remove duplicates
                selected_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            else:
                selected_features = standard_features
            
            # Add RFE selection
            if self.standard_settings.get('enable_rfe_selection', True):
                rfe_features = self.feature_selector.select_features_by_rfe(
                    features=features,
                    targets=targets,
                    n_features=self.standard_settings.get('n_features', 50)
                )
                
                # Combine with selected features
                combined_features = pd.concat([selected_features, rfe_features], axis=1)
                # Remove duplicates
                selected_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            
            # Add mutual information selection
            if self.standard_settings.get('enable_mutual_info_selection', True):
                mi_features = self.feature_selector.select_features_by_mutual_info(
                    features=features,
                    targets=targets,
                    threshold=self.standard_settings.get('mutual_info_threshold', 0.01)
                )
                
                # Combine with selected features
                combined_features = pd.concat([selected_features, mi_features], axis=1)
                # Remove duplicates
                selected_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            
            # Limit features if specified
            max_features = self.standard_settings.get('max_features', 200)
            if len(selected_features.columns) > max_features:
                self.logger.warning(f"Limiting features from {len(selected_features.columns)} to {max_features}")
                # Select top features by variance
                feature_variances = selected_features.var().sort_values(ascending=False)
                top_features = feature_variances.head(max_features).index
                selected_features = selected_features[top_features]
            
            return selected_features
            
        except Exception as e:
            self.logger.exception(f"Error in comprehensive feature selection: {e}")
            raise
    
    def _generate_selection_metadata(self, original_features: pd.DataFrame, selected_features: pd.DataFrame, 
                                   selection_type: str) -> Dict[str, Any]:
        """Generate metadata about feature selection."""
        try:
            metadata = {
                'selection_type': selection_type,
                'original_features': len(original_features.columns),
                'selected_features': len(selected_features.columns),
                'feature_reduction_ratio': len(selected_features.columns) / len(original_features.columns),
                'selected_feature_names': list(selected_features.columns),
                'data_shape': selected_features.shape,
                'created_at': datetime.now().isoformat(),
                'settings_used': self.standard_settings
            }
            
            # Add feature statistics
            if len(selected_features) > 0:
                metadata['feature_statistics'] = {
                    'mean_values': selected_features.mean().to_dict(),
                    'std_values': selected_features.std().to_dict(),
                    'min_values': selected_features.min().to_dict(),
                    'max_values': selected_features.max().to_dict(),
                    'missing_values': selected_features.isnull().sum().to_dict()
                }
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error generating selection metadata: {e}")
            return {'error': str(e)}
    
    def get_feature_selection_summary(self) -> Dict[str, Any]:
        """Get summary of feature selection capabilities."""
        return {
            'config': self.config,
            'standard_settings': self.standard_settings,
            'feature_selector_info': {
                'selector_type': 'Step08AdvancedFeatureSelection',
                'available_methods': [
                    'select_features_by_correlation',
                    'select_features_by_importance',
                    'select_features_by_mrmr',
                    'select_features_by_rfe',
                    'select_features_by_mutual_info'
                ]
            },
            'timestamp': datetime.now().isoformat()
        }


# Simplified feature selection step functions
async def unified_feature_selection_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified feature selection logic using Step08AdvancedFeatureSelection.
    
    Args:
        config: Configuration dictionary
        pipeline_state: Current pipeline state
        
    Returns:
        Feature selection result
    """
    logger.info("🎯 Starting unified feature selection...")
    
    try:
        # Get features and targets from pipeline state
        features = pipeline_state.get('features')
        targets = pipeline_state.get('targets') or pipeline_state.get('labels')
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for feature selection")
        
        # Initialize unified feature selection manager
        selection_manager = UnifiedFeatureSelectionManager(config)
        
        # Determine selection type from configuration
        selection_type = config.get('selection_type', 'comprehensive')
        
        # Select features
        result = await selection_manager.select_features(features, targets, selection_type)
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in unified feature selection: {e}")
        raise


async def basic_feature_selection_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Basic feature selection logic (correlation-based only)."""
    logger.info("🎯 Starting basic feature selection...")
    
    try:
        # Get features and targets from pipeline state
        features = pipeline_state.get('features')
        targets = pipeline_state.get('targets') or pipeline_state.get('labels')
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for feature selection")
        
        # Initialize unified feature selection manager
        selection_manager = UnifiedFeatureSelectionManager(config)
        
        # Select features using basic approach
        result = await selection_manager.select_features(features, targets, 'basic')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in basic feature selection: {e}")
        raise


async def standard_feature_selection_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Standard feature selection logic (correlation + importance)."""
    logger.info("🎯 Starting standard feature selection...")
    
    try:
        # Get features and targets from pipeline state
        features = pipeline_state.get('features')
        targets = pipeline_state.get('targets') or pipeline_state.get('labels')
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for feature selection")
        
        # Initialize unified feature selection manager
        selection_manager = UnifiedFeatureSelectionManager(config)
        
        # Select features using standard approach
        result = await selection_manager.select_features(features, targets, 'standard')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in standard feature selection: {e}")
        raise


async def comprehensive_feature_selection_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive feature selection logic (all methods)."""
    logger.info("🎯 Starting comprehensive feature selection...")
    
    try:
        # Get features and targets from pipeline state
        features = pipeline_state.get('features')
        targets = pipeline_state.get('targets') or pipeline_state.get('labels')
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for feature selection")
        
        # Initialize unified feature selection manager
        selection_manager = UnifiedFeatureSelectionManager(config)
        
        # Select features using comprehensive approach
        result = await selection_manager.select_features(features, targets, 'comprehensive')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in comprehensive feature selection: {e}")
        raise


# Create step functions
unified_feature_selection = create_data_processing_step_function("unified_feature_selection", unified_feature_selection_logic)
basic_feature_selection = create_data_processing_step_function("basic_feature_selection", basic_feature_selection_logic)
standard_feature_selection = create_data_processing_step_function("standard_feature_selection", standard_feature_selection_logic)
comprehensive_feature_selection = create_data_processing_step_function("comprehensive_feature_selection", comprehensive_feature_selection_logic)


class SimplifiedFeatureSelection:
    """
    Simplified feature selection using unified infrastructure.
    
    This replaces multiple feature selection implementations with a unified
    approach using Step08AdvancedFeatureSelection from step08_utilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simplified feature selection."""
        self.config = validate_and_fix_config(config, 'feature_selection')
        self.logger = logger.getChild('SimplifiedFeatureSelection')
        
        # Initialize unified feature selection manager
        self.selection_manager = UnifiedFeatureSelectionManager(self.config)
        
        self.logger.info("🚀 Simplified Feature Selection initialized")
    
    async def select_features(self, features: pd.DataFrame, targets: pd.Series, 
                             selection_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Select features using unified approach.
        
        Args:
            features: Input features
            targets: Target values
            selection_type: Type of selection
            
        Returns:
            Feature selection result
        """
        try:
            self.logger.info(f"🚀 Selecting features using {selection_type} approach...")
            
            # Select features
            result = await self.selection_manager.select_features(features, targets, selection_type)
            
            self.logger.info(f"✅ Feature selection completed: {result['selection_metadata']['selected_features']} features selected")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Feature selection error: {e}")
            raise
    
    def get_feature_selection_summary(self) -> Dict[str, Any]:
        """Get summary of feature selection capabilities."""
        return self.selection_manager.get_feature_selection_summary()


# Backward compatibility wrappers
class AdvancedFeatureSelection(SimplifiedFeatureSelection):
    """Backward compatibility wrapper for AdvancedFeatureSelection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("🔄 Using backward compatibility wrapper for AdvancedFeatureSelection")


class FeatureSelectionStep(SimplifiedFeatureSelection):
    """Backward compatibility wrapper for FeatureSelectionStep."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("🔄 Using backward compatibility wrapper for FeatureSelectionStep")


# Example usage and testing
async def example_feature_selection():
    """Example of using the unified feature selection."""
    
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 100
    
    # Create features with some signal
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some signal to first 20 features
    for i in range(20):
        features[f'feature_{i}'] += np.random.randn(n_samples) * 0.5
    
    # Create targets based on first 20 features
    targets = pd.Series(
        (features.iloc[:, :20].sum(axis=1) > 0).astype(int),
        name='target'
    )
    
    # Configuration for different selection types
    configs = [
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'selection_type': 'basic',
            'feature_selection_config': {
                'enable_correlation_selection': True,
                'correlation_threshold': 0.9
            }
        },
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'selection_type': 'standard',
            'feature_selection_config': {
                'enable_correlation_selection': True,
                'enable_importance_selection': True,
                'n_features': 30
            }
        },
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'selection_type': 'comprehensive',
            'feature_selection_config': {
                'enable_mrmr_selection': True,
                'enable_importance_selection': True,
                'enable_rfe_selection': True,
                'enable_correlation_selection': True,
                'enable_mutual_info_selection': True,
                'n_features': 50,
                'max_features': 100
            }
        }
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n=== Testing Feature Selection Type {i+1}: {config['selection_type']} ===")
        
        # Create simplified feature selection
        feature_selector = SimplifiedFeatureSelection(config)
        
        # Select features
        result = await feature_selector.select_features(features, targets, config['selection_type'])
        
        # Get summary
        summary = feature_selector.get_feature_selection_summary()
        
        print(f"Selection type: {result['selection_type']}")
        print(f"Original features: {result['selection_metadata']['original_features']}")
        print(f"Selected features: {result['selection_metadata']['selected_features']}")
        print(f"Reduction ratio: {result['selection_metadata']['feature_reduction_ratio']:.3f}")
        print(f"Data shape: {result['selected_features'].shape}")
        
        results.append((result, summary))
    
    return results


# Main execution
async def main():
    """Main execution function."""
    try:
        results = await example_feature_selection()
        print("✅ Feature selection example completed successfully")
        return results
    except Exception as e:
        logger.exception(f"Feature selection example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())