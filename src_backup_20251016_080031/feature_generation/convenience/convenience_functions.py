"""
Convenience Functions for Feature Generation

This module provides convenient functions for common feature generation tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import json

from ..core.feature_bank import get_global_feature_bank
from ..core.feature_generator import FeatureCategory

logger = logging.getLogger(__name__)

def generate_features_by_category(data: pd.DataFrame,
                                categories: List[Union[str, FeatureCategory]],
                                lookback_optimization: bool = False,
                                target_column: Optional[str] = None,
                                **kwargs) -> pd.DataFrame:
    """
    Generate features by category using the feature bank.
    
    Args:
        data: Input data DataFrame
        categories: List of categories to generate features for
        lookback_optimization: Whether to optimize lookback periods
        target_column: Target column for lookback optimization
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with generated features
    """
    bank = get_global_feature_bank()
    
    return bank.generate_features(
        data=data,
        categories=categories,
        lookback_optimization=lookback_optimization,
        target_column=target_column,
        **kwargs
    )

def generate_all_features(data: pd.DataFrame,
                         lookback_optimization: bool = False,
                         target_column: Optional[str] = None,
                         **kwargs) -> pd.DataFrame:
    """
    Generate all available features.
    
    Args:
        data: Input data DataFrame
        lookback_optimization: Whether to optimize lookback periods
        target_column: Target column for lookback optimization
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with all generated features
    """
    bank = get_global_feature_bank()
    
    return bank.generate_features(
        data=data,
        lookback_optimization=lookback_optimization,
        target_column=target_column,
        **kwargs
    )

def get_feature_summary(category: Optional[Union[str, FeatureCategory]] = None) -> Dict[str, Any]:
    """
    Get a summary of available features.
    
    Args:
        category: Optional category filter
        
    Returns:
        Dictionary with feature summary
    """
    bank = get_global_feature_bank()
    
    if category is not None:
        if isinstance(category, str):
            try:
                category = FeatureCategory(category)
            except ValueError:
                logger.warning(f"Invalid category: {category}")
                return {}
        
        generators = bank.get_generators_by_category(category)
        return {
            'category': category.value,
            'total_features': len(generators),
            'features': [gen.config.name for gen in generators]
        }
    
    return bank.get_feature_summary()

def validate_feature_data(data: pd.DataFrame,
                         feature_names: Optional[List[str]] = None,
                         categories: Optional[List[Union[str, FeatureCategory]]] = None) -> Dict[str, Any]:
    """
    Validate that data has required columns for specified features.
    
    Args:
        data: Input data DataFrame
        feature_names: Optional list of specific feature names
        categories: Optional list of categories
        
    Returns:
        Dictionary with validation results
    """
    bank = get_global_feature_bank()
    
    # Determine which features to validate
    if feature_names:
        features_to_validate = feature_names
    elif categories:
        features_to_validate = []
        for category in categories:
            if isinstance(category, str):
                try:
                    category = FeatureCategory(category)
                except ValueError:
                    continue
            generators = bank.get_generators_by_category(category)
            features_to_validate.extend([gen.config.name for gen in generators])
    else:
        # Validate all features
        features_to_validate = bank.list_features()
    
    # Validate features
    from ..core.factory import validate_feature_requirements
    return validate_feature_requirements(
        data_columns=list(data.columns),
        feature_names=features_to_validate,
        bank=bank
    )

def export_feature_config(output_file: str,
                         categories: Optional[List[Union[str, FeatureCategory]]] = None,
                         include_parameters: bool = True) -> None:
    """
    Export feature configuration to a file.
    
    Args:
        output_file: Output file path
        categories: Optional list of categories to export
        include_parameters: Whether to include feature parameters
    """
    bank = get_global_feature_bank()
    
    config = {
        'feature_bank_config': {
            'enable_matrix_operations': bank.config.enable_matrix_operations,
            'enable_gpu_acceleration': bank.config.enable_gpu_acceleration,
            'enable_lookback_optimization': bank.config.enable_lookback_optimization,
            'enable_parallel_processing': bank.config.enable_parallel_processing,
            'max_workers': bank.config.max_workers,
            'default_lookback': bank.config.default_lookback
        },
        'features': {}
    }
    
    # Get features to export
    if categories:
        features_to_export = []
        for category in categories:
            if isinstance(category, str):
                try:
                    category = FeatureCategory(category)
                except ValueError:
                    continue
            generators = bank.get_generators_by_category(category)
            features_to_export.extend(generators)
    else:
        features_to_export = bank.registry.get_all()
    
    # Export feature configurations
    for generator in features_to_export:
        feature_config = {
            'name': generator.config.name,
            'category': generator.config.category.value,
            'description': generator.config.description,
            'required_columns': generator.config.required_columns,
            'optional_columns': generator.config.optional_columns,
            'default_lookback': generator.config.default_lookback,
            'min_lookback': generator.config.min_lookback,
            'max_lookback': generator.config.max_lookback,
            'dependencies': generator.config.dependencies,
            'matrix_optimized': generator.config.matrix_optimized,
            'gpu_accelerated': generator.config.gpu_accelerated
        }
        
        if include_parameters:
            feature_config['parameters'] = generator.config.parameters
        
        config['features'][generator.config.name] = feature_config
    
    # Write to file
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Feature configuration exported to {output_file}")

def import_feature_config(config_file: str) -> Dict[str, Any]:
    """
    Import feature configuration from a file.
    
    Args:
        config_file: Configuration file path
        
    Returns:
        Imported configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Feature configuration imported from {config_file}")
    return config

def create_feature_pipeline(categories: List[Union[str, FeatureCategory]],
                          lookback_optimization: bool = False,
                          target_column: Optional[str] = None) -> callable:
    """
    Create a feature generation pipeline function.
    
    Args:
        categories: List of categories to include
        lookback_optimization: Whether to optimize lookback periods
        target_column: Target column for optimization
        
    Returns:
        Pipeline function that takes data and returns features
    """
    def pipeline(data: pd.DataFrame) -> pd.DataFrame:
        return generate_features_by_category(
            data=data,
            categories=categories,
            lookback_optimization=lookback_optimization,
            target_column=target_column
        )
    
    return pipeline

def get_feature_categories() -> List[str]:
    """
    Get list of available feature categories.
    
    Returns:
        List of category names
    """
    bank = get_global_feature_bank()
    return [cat.value for cat in bank.list_categories()]

def get_features_by_category(category: Union[str, FeatureCategory]) -> List[str]:
    """
    Get list of features for a specific category.
    
    Args:
        category: Feature category
        
    Returns:
        List of feature names
    """
    bank = get_global_feature_bank()
    
    if isinstance(category, str):
        try:
            category = FeatureCategory(category)
        except ValueError:
            logger.warning(f"Invalid category: {category}")
            return []
    
    return bank.list_features(category)

def search_features(query: str, category: Optional[Union[str, FeatureCategory]] = None) -> List[str]:
    """
    Search for features by name or description.
    
    Args:
        query: Search query
        category: Optional category filter
        
    Returns:
        List of matching feature names
    """
    from ..core.factory import search_features as core_search_features
    return core_search_features(query, category)

def get_feature_info(feature_name: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a feature.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Dictionary with feature information or None if not found
    """
    from ..core.factory import get_feature_info as core_get_feature_info
    return core_get_feature_info(feature_name)
