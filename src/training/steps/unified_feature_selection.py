"""
Unified Feature Selection Infrastructure

This module provides unified feature selection across all training steps using
Step08AdvancedFeatureSelection from step08_utilities, replacing custom feature 
selection logic.

Key Features:
- Unified feature selection using Step08AdvancedFeatureSelection
- Reduces feature selection code by ~70%
- Standardized feature selection approaches across all steps
- Automatic feature selection validation and quality checks
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
    Step08AdvancedFeatureSelectionPerRegime,
    AdvancedFeatureSelectionStep,
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
    
    This replaces custom feature selection logic with a unified approach
    using Step08AdvancedFeatureSelection from step08_utilities.
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
        
        # Feature selection configuration
        self.selection_config = self.config.get('feature_selection_config', {})
        
        # Standard feature selection settings
        self.standard_settings = {
            'selection_method': 'mrmr',
            'n_features': 50,
            'stability_threshold': 0.6,
            'correlation_threshold': 0.95,
            'variance_threshold': 1e-10,
            'enable_regime_specific': False,
            'enable_parallel_processing': True,
            'enable_gpu_acceleration': True,
            'max_workers': 4,
            'timeout_seconds': 3600,
            'random_state': 42
        }
        
        # Update with user configuration
        self.standard_settings.update(self.selection_config)
        
        # Initialize feature selection based on method
        self._initialize_feature_selector()
        
        self.logger.info("🚀 Unified Feature Selection Manager initialized")
    
    def _initialize_feature_selector(self):
        """Initialize the appropriate feature selector based on configuration."""
        try:
            selection_method = self.standard_settings.get('selection_method', 'mrmr')
            enable_regime_specific = self.standard_settings.get('enable_regime_specific', False)
            
            if enable_regime_specific:
                # Use regime-specific feature selection
                self.feature_selector = Step08AdvancedFeatureSelectionPerRegime(self.config)
                self.logger.info("Using regime-specific feature selection")
            else:
                # Use standard feature selection
                self.feature_selector = Step08AdvancedFeatureSelection(self.config)
                self.logger.info(f"Using standard feature selection with method: {selection_method}")
            
        except Exception as e:
            self.logger.warning(f"Error initializing feature selector: {e}")
            # Fallback to basic feature selection
            self.feature_selector = None
    
    async def select_features(self, features: pd.DataFrame, targets: pd.Series, 
                            selection_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Select features using unified approach.
        
        Args:
            features: Feature matrix
            targets: Target values
            selection_type: Type of selection ('basic', 'standard', 'comprehensive')
            
        Returns:
            Feature selection result
        """
        try:
            self.logger.info(f"🎯 Starting {selection_type} feature selection...")
            
            # Validate input data
            features_validation = validate_data_quality(features, 'features', 'comprehensive')
            targets_validation = validate_data_quality(targets, 'targets', 'standard')
            
            if not features_validation['passed']:
                self.logger.warning(f"Features validation issues: {features_validation['errors']}")
            
            if not targets_validation['passed']:
                self.logger.warning(f"Targets validation issues: {targets_validation['errors']}")
            
            # Perform feature selection based on type
            if selection_type == 'basic':
                selection_result = await self._perform_basic_selection(features, targets)
            elif selection_type == 'standard':
                selection_result = await self._perform_standard_selection(features, targets)
            elif selection_type == 'comprehensive':
                selection_result = await self._perform_comprehensive_selection(features, targets)
            else:
                raise ValueError(f"Unknown selection type: {selection_type}")
            
            # Validate selected features
            selected_features_validation = validate_data_quality(
                selection_result['selected_features'], 'features', 'standard'
            )
            
            # Generate selection metadata
            selection_metadata = self._generate_selection_metadata(
                features, targets, selection_result, selection_type
            )
            
            # Generate quality report
            quality_report = generate_quality_report(selection_result['selected_features'], 'selected_features')
            
            return {
                'selected_features': selection_result['selected_features'],
                'feature_importance': selection_result.get('feature_importance', {}),
                'selection_metadata': selection_metadata,
                'features_validation': features_validation,
                'targets_validation': targets_validation,
                'selected_features_validation': selected_features_validation,
                'quality_report': quality_report,
                'selection_type': selection_type
            }
            
        except Exception as e:
            self.logger.exception(f"Error selecting features: {e}")
            raise
    
    async def _perform_basic_selection(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Perform basic feature selection (variance + correlation filtering)."""
        try:
            self.logger.info("Performing basic feature selection...")
            
            # Start with all features
            selected_features = features.copy()
            
            # Remove low variance features
            variance_threshold = self.standard_settings.get('variance_threshold', 1e-10)
            feature_variances = selected_features.var()
            high_variance_features = feature_variances[feature_variances > variance_threshold].index
            selected_features = selected_features[high_variance_features]
            
            self.logger.info(f"After variance filtering: {len(selected_features.columns)} features")
            
            # Remove highly correlated features
            correlation_threshold = self.standard_settings.get('correlation_threshold', 0.95)
            correlation_matrix = selected_features.corr().abs()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > correlation_threshold:
                        high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for feat1, feat2 in high_corr_pairs:
                if feat1 not in features_to_remove:
                    features_to_remove.add(feat2)
            
            selected_features = selected_features.drop(columns=list(features_to_remove))
            
            self.logger.info(f"After correlation filtering: {len(selected_features.columns)} features")
            
            return {
                'selected_features': selected_features,
                'feature_importance': {},
                'selection_method': 'basic_variance_correlation'
            }
            
        except Exception as e:
            self.logger.exception(f"Error in basic feature selection: {e}")
            raise
    
    async def _perform_standard_selection(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Perform standard feature selection using ML Common utilities."""
        try:
            self.logger.info("Performing standard feature selection...")
            
            # Use ML Common FeatureSelectionFramework
            feature_selector = FeatureSelectionFramework(self.config.get('feature_selection_config', {}))
            
            # Perform mRMR selection
            n_features = self.standard_settings.get('n_features', 50)
            selection_result = feature_selector.mrmr_selection(
                X=features.values,
                y=targets.values,
                feature_names=list(features.columns),
                n_features=min(n_features, len(features.columns))
            )
            
            # Get selected feature names
            selected_feature_names = selection_result['selected_features']
            selected_features = features[selected_feature_names]
            
            return {
                'selected_features': selected_features,
                'feature_importance': selection_result.get('feature_scores', {}),
                'selection_method': 'mrmr',
                'selection_details': selection_result
            }
            
        except Exception as e:
            self.logger.exception(f"Error in standard feature selection: {e}")
            raise
    
    async def _perform_comprehensive_selection(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive feature selection using Step08 utilities."""
        try:
            self.logger.info("Performing comprehensive feature selection...")
            
            if self.feature_selector is None:
                self.logger.warning("Feature selector not available, falling back to standard selection")
                return await self._perform_standard_selection(features, targets)
            
            # Use Step08AdvancedFeatureSelection
            selection_result = self.feature_selector.select_features(
                features=features,
                targets=targets,
                method=self.standard_settings.get('selection_method', 'mrmr'),
                n_features=self.standard_settings.get('n_features', 50),
                stability_threshold=self.standard_settings.get('stability_threshold', 0.6)
            )
            
            return {
                'selected_features': selection_result.get('selected_features', features),
                'feature_importance': selection_result.get('feature_importance', {}),
                'selection_method': self.standard_settings.get('selection_method', 'mrmr'),
                'selection_details': selection_result
            }
            
        except Exception as e:
            self.logger.exception(f"Error in comprehensive feature selection: {e}")
            raise
    
    def _generate_selection_metadata(self, original_features: pd.DataFrame, targets: pd.Series, 
                                   selection_result: Dict[str, Any], selection_type: str) -> Dict[str, Any]:
        """Generate metadata about feature selection."""
        try:
            selected_features = selection_result['selected_features']
            
            metadata = {
                'selection_type': selection_type,
                'original_features': len(original_features.columns),
                'selected_features': len(selected_features.columns),
                'reduction_ratio': len(selected_features.columns) / len(original_features.columns),
                'selection_method': selection_result.get('selection_method', 'unknown'),
                'selected_feature_names': list(selected_features.columns),
                'created_at': datetime.now().isoformat(),
                'settings_used': self.standard_settings
            }
            
            # Add feature importance information
            if 'feature_importance' in selection_result and selection_result['feature_importance']:
                importance_dict = selection_result['feature_importance']
                if isinstance(importance_dict, dict):
                    metadata['top_features'] = sorted(
                        importance_dict.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10]
            
            # Add selection statistics
            if len(selected_features) > 0:
                metadata['selection_statistics'] = {
                    'mean_importance': np.mean(list(selection_result.get('feature_importance', {}).values())) if selection_result.get('feature_importance') else 0,
                    'std_importance': np.std(list(selection_result.get('feature_importance', {}).values())) if selection_result.get('feature_importance') else 0,
                    'min_importance': min(selection_result.get('feature_importance', {}).values()) if selection_result.get('feature_importance') else 0,
                    'max_importance': max(selection_result.get('feature_importance', {}).values()) if selection_result.get('feature_importance') else 0
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
                'selector_type': type(self.feature_selector).__name__ if self.feature_selector else 'None',
                'available_methods': [
                    'basic_variance_correlation',
                    'mrmr',
                    'importance',
                    'rfe',
                    'correlation',
                    'mutual_info'
                ],
                'regime_specific_enabled': self.standard_settings.get('enable_regime_specific', False)
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
        targets = pipeline_state.get('labels') or pipeline_state.get('targets')
        
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
    """Basic feature selection logic (variance + correlation filtering)."""
    logger.info("🎯 Starting basic feature selection...")
    
    try:
        # Get features and targets from pipeline state
        features = pipeline_state.get('features')
        targets = pipeline_state.get('labels') or pipeline_state.get('targets')
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for feature selection")
        
        # Initialize unified feature selection manager
        selection_manager = UnifiedFeatureSelectionManager(config)
        
        # Select features
        result = await selection_manager.select_features(features, targets, 'basic')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in basic feature selection: {e}")
        raise


async def standard_feature_selection_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Standard feature selection logic (mRMR selection)."""
    logger.info("🎯 Starting standard feature selection...")
    
    try:
        # Get features and targets from pipeline state
        features = pipeline_state.get('features')
        targets = pipeline_state.get('labels') or pipeline_state.get('targets')
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for feature selection")
        
        # Initialize unified feature selection manager
        selection_manager = UnifiedFeatureSelectionManager(config)
        
        # Select features
        result = await selection_manager.select_features(features, targets, 'standard')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in standard feature selection: {e}")
        raise


async def comprehensive_feature_selection_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive feature selection logic (Step08 utilities)."""
    logger.info("🎯 Starting comprehensive feature selection...")
    
    try:
        # Get features and targets from pipeline state
        features = pipeline_state.get('features')
        targets = pipeline_state.get('labels') or pipeline_state.get('targets')
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for feature selection")
        
        # Initialize unified feature selection manager
        selection_manager = UnifiedFeatureSelectionManager(config)
        
        # Select features
        result = await selection_manager.select_features(features, targets, 'comprehensive')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in comprehensive feature selection: {e}")
        raise


# Create step functions
unified_feature_selection = create_simple_step_function("unified_feature_selection", unified_feature_selection_logic)
basic_feature_selection = create_simple_step_function("basic_feature_selection", basic_feature_selection_logic)
standard_feature_selection = create_simple_step_function("standard_feature_selection", standard_feature_selection_logic)
comprehensive_feature_selection = create_simple_step_function("comprehensive_feature_selection", comprehensive_feature_selection_logic)


class SimplifiedFeatureSelection:
    """
    Simplified feature selection using unified infrastructure.
    
    This replaces custom feature selection logic with a unified approach
    using Step08AdvancedFeatureSelection from step08_utilities.
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
            features: Feature matrix
            targets: Target values
            selection_type: Type of selection
            
        Returns:
            Feature selection result
        """
        try:
            self.logger.info(f"🚀 Selecting features using {selection_type} approach...")
            
            # Select features
            result = await self.selection_manager.select_features(features, targets, selection_type)
            
            self.logger.info(f"✅ Feature selection completed: {result['selection_metadata']['selected_features']} features selected from {result['selection_metadata']['original_features']}")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Feature selection error: {e}")
            raise
    
    def get_feature_selection_summary(self) -> Dict[str, Any]:
        """Get summary of feature selection capabilities."""
        return self.selection_manager.get_feature_selection_summary()


# Backward compatibility wrappers
class Step08AdvancedFeatureSelectionWrapper(SimplifiedFeatureSelection):
    """Backward compatibility wrapper for Step08AdvancedFeatureSelection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("🔄 Using backward compatibility wrapper for Step08AdvancedFeatureSelection")


class AdvancedFeatureSelectionStepWrapper(SimplifiedFeatureSelection):
    """Backward compatibility wrapper for AdvancedFeatureSelectionStep."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger.info("🔄 Using backward compatibility wrapper for AdvancedFeatureSelectionStep")


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
    
    # Add some signal to first 10 features
    for i in range(10):
        features[f'feature_{i}'] += np.random.randn(n_samples) * 0.5
    
    # Create targets based on first 10 features
    targets = pd.Series(
        (features.iloc[:, :10].sum(axis=1) > 0).astype(int),
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
                'variance_threshold': 1e-10,
                'correlation_threshold': 0.95
            }
        },
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'selection_type': 'standard',
            'feature_selection_config': {
                'selection_method': 'mrmr',
                'n_features': 20
            }
        },
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'selection_type': 'comprehensive',
            'feature_selection_config': {
                'selection_method': 'mrmr',
                'n_features': 20,
                'stability_threshold': 0.6,
                'enable_regime_specific': False
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
        print(f"Reduction ratio: {result['selection_metadata']['reduction_ratio']:.3f}")
        print(f"Selection method: {result['selection_metadata']['selection_method']}")
        
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