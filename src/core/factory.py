"""
Factory Functions for Abstract Base Classes

This module provides factory functions for easy instantiation of all abstract base classes
and their concrete implementations. These factories handle configuration, validation,
and provide a consistent interface for creating instances.

Key Features:
- Type-safe factory functions for all base classes
- Automatic configuration validation and defaults
- Integration with existing utilities and logging
- Support for both concrete and custom implementations
- Comprehensive error handling and validation
"""

from typing import Any, Dict, List, Optional, Type, Union
import logging
from pathlib import Path

# Import base classes
from src.core.abstract_base_classes import (
    BaseValidator, BaseTrainingStep, BaseClusteringAlgorithm,
    MultiOutputModel, BasePatternDiscoverer, BaseLabelingStrategy,
    ValidationLevel, TrainingStatus, ClusteringAlgorithm, PatternType, LabelingStrategy
)

# Import concrete implementations
from src.core.concrete_implementations import (
    DataValidator, MLTrainingStep, KMeansClustering,
    MultiOutputRandomForest, MomentumPatternDiscoverer, ProfitBasedLabeling
)

# Import existing implementations
from src.utils.base_validator import BaseValidator as ExistingBaseValidator
from src.utils.ml_common.training.base_training_step import BaseTrainingStep as ExistingBaseTrainingStep
from src.training.steps.market_analysis.components.clustering_algorithms import BaseClusteringAlgorithm as ExistingBaseClusteringAlgorithm
from src.utils.ml_common.models.multi_output_models import MultiOutputModel as ExistingMultiOutputModel
from src.research.price_patterns.pattern_discovery_framework import BasePatternDiscoverer as ExistingBasePatternDiscoverer
from src.research.profit_labeling.ensemble_labeling_system import BaseLabelingStrategy as ExistingBaseLabelingStrategy

# Setup logging
logger = logging.getLogger(__name__)

class BaseClassFactory:
    """
    Factory class for creating instances of abstract base classes.
    
    This factory provides a centralized way to create instances of all base classes
    with proper configuration, validation, and error handling.
    """
    
    @staticmethod
    def create_validator(
        name: str,
        validator_type: str = "data",
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        config: Optional[Dict[str, Any]] = None,
        use_production: bool = True
    ) -> Union[BaseValidator, ExistingBaseValidator]:
        """
        Create a validator instance.
        
        Args:
            name: Name of the validator
            validator_type: Type of validator ("data", "model", "config")
            validation_level: Validation level to use
            config: Configuration dictionary
            use_production: Whether to use production-ready base class
            
        Returns:
            Validator instance
        """
        try:
            config = config or {}
            
            if use_production:
                if validator_type == "data":
                    return DataValidator(
                        name=name,
                        validation_level=validation_level,
                        config=config
                    )
                else:
                    # Use production base class for custom implementations
                    return BaseValidator(
                        name=name,
                        validation_level=validation_level,
                        config=config
                    )
            else:
                # Use existing base class for backward compatibility
                return ExistingBaseValidator(
                    step_name=name,
                    config=config
                )
                
        except Exception as e:
            logger.error(f"Failed to create validator {name}: {e}")
            raise

    @staticmethod
    def create_training_step(
        name: str,
        model_type: str = "random_forest",
        config: Optional[Dict[str, Any]] = None,
        use_production: bool = True
    ) -> Union[BaseTrainingStep, ExistingBaseTrainingStep]:
        """
        Create a training step instance.
        
        Args:
            name: Name of the training step
            model_type: Type of model to use
            config: Configuration dictionary
            use_production: Whether to use production-ready base class
            
        Returns:
            Training step instance
        """
        try:
            config = config or {}
            
            if use_production:
                return MLTrainingStep(
                    name=name,
                    model_type=model_type,
                    config=config
                )
            else:
                # Use existing base class for backward compatibility
                from src.utils.ml_common.config.base_training_config import BaseTrainingConfig
                training_config = BaseTrainingConfig(**config)
                return ExistingBaseTrainingStep(training_config)
                
        except Exception as e:
            logger.error(f"Failed to create training step {name}: {e}")
            raise

    @staticmethod
    def create_clustering_algorithm(
        name: str,
        algorithm: ClusteringAlgorithm = ClusteringAlgorithm.KMEANS,
        n_clusters: int = 5,
        config: Optional[Dict[str, Any]] = None,
        use_production: bool = True
    ) -> Union[BaseClusteringAlgorithm, ExistingBaseClusteringAlgorithm]:
        """
        Create a clustering algorithm instance.
        
        Args:
            name: Name of the clustering algorithm
            algorithm: Type of clustering algorithm
            n_clusters: Number of clusters
            config: Configuration dictionary
            use_production: Whether to use production-ready base class
            
        Returns:
            Clustering algorithm instance
        """
        try:
            config = config or {}
            
            if use_production:
                if algorithm == ClusteringAlgorithm.KMEANS:
                    return KMeansClustering(
                        name=name,
                        n_clusters=n_clusters,
                        config=config
                    )
                else:
                    # Use production base class for custom implementations
                    return BaseClusteringAlgorithm(
                        name=name,
                        algorithm=algorithm,
                        config=config
                    )
            else:
                # Use existing base class for backward compatibility
                return ExistingBaseClusteringAlgorithm(
                    name=name,
                    algorithm=algorithm.value,
                    config=config
                )
                
        except Exception as e:
            logger.error(f"Failed to create clustering algorithm {name}: {e}")
            raise

    @staticmethod
    def create_multi_output_model(
        name: str,
        n_outputs: int,
        output_names: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        use_production: bool = True
    ) -> Union[MultiOutputModel, ExistingMultiOutputModel]:
        """
        Create a multi-output model instance.
        
        Args:
            name: Name of the model
            n_outputs: Number of outputs
            output_names: Names of the outputs
            config: Configuration dictionary
            use_production: Whether to use production-ready base class
            
        Returns:
            Multi-output model instance
        """
        try:
            config = config or {}
            
            if use_production:
                return MultiOutputRandomForest(
                    name=name,
                    n_outputs=n_outputs,
                    output_names=output_names,
                    config=config
                )
            else:
                # Use existing base class for backward compatibility
                return ExistingMultiOutputModel(
                    name=name,
                    n_outputs=n_outputs,
                    output_names=output_names,
                    config=config
                )
                
        except Exception as e:
            logger.error(f"Failed to create multi-output model {name}: {e}")
            raise

    @staticmethod
    def create_pattern_discoverer(
        name: str,
        pattern_type: PatternType = PatternType.MOMENTUM,
        config: Optional[Dict[str, Any]] = None,
        use_production: bool = True
    ) -> Union[BasePatternDiscoverer, ExistingBasePatternDiscoverer]:
        """
        Create a pattern discoverer instance.
        
        Args:
            name: Name of the pattern discoverer
            pattern_type: Type of pattern to discover
            config: Configuration dictionary
            use_production: Whether to use production-ready base class
            
        Returns:
            Pattern discoverer instance
        """
        try:
            config = config or {}
            
            if use_production:
                if pattern_type == PatternType.MOMENTUM:
                    return MomentumPatternDiscoverer(
                        name=name,
                        config=config
                    )
                else:
                    # Use production base class for custom implementations
                    return BasePatternDiscoverer(
                        name=name,
                        pattern_type=pattern_type,
                        config=config
                    )
            else:
                # Use existing base class for backward compatibility
                return ExistingBasePatternDiscoverer(
                    name=name,
                    pattern_type=pattern_type,
                    config=config
                )
                
        except Exception as e:
            logger.error(f"Failed to create pattern discoverer {name}: {e}")
            raise

    @staticmethod
    def create_labeling_strategy(
        name: str,
        strategy: LabelingStrategy = LabelingStrategy.PROFIT_BASED,
        config: Optional[Dict[str, Any]] = None,
        use_production: bool = True
    ) -> Union[BaseLabelingStrategy, ExistingBaseLabelingStrategy]:
        """
        Create a labeling strategy instance.
        
        Args:
            name: Name of the labeling strategy
            strategy: Type of labeling strategy
            config: Configuration dictionary
            use_production: Whether to use production-ready base class
            
        Returns:
            Labeling strategy instance
        """
        try:
            config = config or {}
            
            if use_production:
                if strategy == LabelingStrategy.PROFIT_BASED:
                    return ProfitBasedLabeling(
                        name=name,
                        config=config
                    )
                else:
                    # Use production base class for custom implementations
                    return BaseLabelingStrategy(
                        name=name,
                        strategy=strategy,
                        config=config
                    )
            else:
                # Use existing base class for backward compatibility
                return ExistingBaseLabelingStrategy(
                    name=name,
                    strategy=strategy,
                    config=config
                )
                
        except Exception as e:
            logger.error(f"Failed to create labeling strategy {name}: {e}")
            raise

# Convenience functions for direct instantiation
def create_validator(name: str, **kwargs) -> Union[BaseValidator, ExistingBaseValidator]:
    """Create a validator instance."""
    return BaseClassFactory.create_validator(name, **kwargs)

def create_training_step(name: str, **kwargs) -> Union[BaseTrainingStep, ExistingBaseTrainingStep]:
    """Create a training step instance."""
    return BaseClassFactory.create_training_step(name, **kwargs)

def create_clustering_algorithm(name: str, **kwargs) -> Union[BaseClusteringAlgorithm, ExistingBaseClusteringAlgorithm]:
    """Create a clustering algorithm instance."""
    return BaseClassFactory.create_clustering_algorithm(name, **kwargs)

def create_multi_output_model(name: str, **kwargs) -> Union[MultiOutputModel, ExistingMultiOutputModel]:
    """Create a multi-output model instance."""
    return BaseClassFactory.create_multi_output_model(name, **kwargs)

def create_pattern_discoverer(name: str, **kwargs) -> Union[BasePatternDiscoverer, ExistingBasePatternDiscoverer]:
    """Create a pattern discoverer instance."""
    return BaseClassFactory.create_pattern_discoverer(name, **kwargs)

def create_labeling_strategy(name: str, **kwargs) -> Union[BaseLabelingStrategy, ExistingBaseLabelingStrategy]:
    """Create a labeling strategy instance."""
    return BaseClassFactory.create_labeling_strategy(name, **kwargs)

# Configuration presets for common use cases
class ConfigurationPresets:
    """Configuration presets for common use cases."""
    
    @staticmethod
    def get_production_config() -> Dict[str, Any]:
        """Get production configuration preset."""
        return {
            'validation_level': ValidationLevel.PRODUCTION,
            'enable_logging': True,
            'enable_metrics': True,
            'enable_optimization': True
        }
    
    @staticmethod
    def get_development_config() -> Dict[str, Any]:
        """Get development configuration preset."""
        return {
            'validation_level': ValidationLevel.STANDARD,
            'enable_logging': True,
            'enable_metrics': True,
            'enable_optimization': False
        }
    
    @staticmethod
    def get_testing_config() -> Dict[str, Any]:
        """Get testing configuration preset."""
        return {
            'validation_level': ValidationLevel.BASIC,
            'enable_logging': False,
            'enable_metrics': False,
            'enable_optimization': False
        }
    
    @staticmethod
    def get_ml_pipeline_config() -> Dict[str, Any]:
        """Get ML pipeline configuration preset."""
        return {
            'model_type': 'random_forest',
            'n_estimators': 200,
            'max_depth': 10,
            'scale_features': True,
            'enable_early_stopping': True
        }
    
    @staticmethod
    def get_clustering_config() -> Dict[str, Any]:
        """Get clustering configuration preset."""
        return {
            'n_clusters': 5,
            'random_state': 42,
            'n_init': 10,
            'max_iter': 300
        }
    
    @staticmethod
    def get_pattern_discovery_config() -> Dict[str, Any]:
        """Get pattern discovery configuration preset."""
        return {
            'lookback_period': 20,
            'momentum_threshold': 0.03,
            'confidence_threshold': 0.7,
            'frequency_threshold': 0.1
        }
    
    @staticmethod
    def get_labeling_config() -> Dict[str, Any]:
        """Get labeling configuration preset."""
        return {
            'profit_threshold': 0.02,
            'lookforward_period': 5,
            'min_confidence': 0.6,
            'max_confidence': 1.0
        }

# Example usage and integration
def create_complete_pipeline(
    pipeline_name: str,
    config_preset: str = "production",
    use_production: bool = True
) -> Dict[str, Any]:
    """
    Create a complete ML pipeline with all components.
    
    Args:
        pipeline_name: Name of the pipeline
        config_preset: Configuration preset to use
        use_production: Whether to use production-ready base classes
        
    Returns:
        Dictionary containing all pipeline components
    """
    try:
        # Get configuration preset
        if config_preset == "production":
            base_config = ConfigurationPresets.get_production_config()
        elif config_preset == "development":
            base_config = ConfigurationPresets.get_development_config()
        elif config_preset == "testing":
            base_config = ConfigurationPresets.get_testing_config()
        else:
            base_config = ConfigurationPresets.get_development_config()
        
        # Create all components
        pipeline = {
            'name': pipeline_name,
            'validator': create_validator(
                f"{pipeline_name}_validator",
                use_production=use_production,
                **base_config
            ),
            'training_step': create_training_step(
                f"{pipeline_name}_training",
                use_production=use_production,
                **ConfigurationPresets.get_ml_pipeline_config()
            ),
            'clustering': create_clustering_algorithm(
                f"{pipeline_name}_clustering",
                use_production=use_production,
                **ConfigurationPresets.get_clustering_config()
            ),
            'multi_output_model': create_multi_output_model(
                f"{pipeline_name}_multi_output",
                n_outputs=3,
                use_production=use_production,
                **ConfigurationPresets.get_ml_pipeline_config()
            ),
            'pattern_discoverer': create_pattern_discoverer(
                f"{pipeline_name}_pattern_discoverer",
                use_production=use_production,
                **ConfigurationPresets.get_pattern_discovery_config()
            ),
            'labeling_strategy': create_labeling_strategy(
                f"{pipeline_name}_labeling",
                use_production=use_production,
                **ConfigurationPresets.get_labeling_config()
            )
        }
        
        logger.info(f"Created complete pipeline: {pipeline_name}")
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to create complete pipeline {pipeline_name}: {e}")
        raise

# Export all factory functions and classes
__all__ = [
    'BaseClassFactory',
    'create_validator',
    'create_training_step', 
    'create_clustering_algorithm',
    'create_multi_output_model',
    'create_pattern_discoverer',
    'create_labeling_strategy',
    'ConfigurationPresets',
    'create_complete_pipeline'
]