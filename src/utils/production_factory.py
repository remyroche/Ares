"""
Production Factory

This module provides factory functions and classes for easy creation and configuration
of all implemented abstract base classes and concrete implementations.
"""

from typing import Dict, Any, List, Optional, Union, Type
import logging
from dataclasses import dataclass

# Import our implemented classes
from src.utils.base_validator import (
    BaseValidator, DataValidator, ModelValidator, ConfigValidator
)
from src.utils.standalone_early_stopping import (
    EarlyStoppingStrategy, AdaptivePatienceStrategy, ConvergenceBasedStrategy,
    PerformanceBasedStrategy, TimeBasedStrategy, TrialBasedStrategy, CompositeStrategy,
    EarlyStoppingConfig
)

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Configuration for production ML system."""
    
    # Validation settings
    enable_data_validation: bool = True
    enable_model_validation: bool = True
    enable_config_validation: bool = True
    
    # Early stopping settings
    enable_early_stopping: bool = True
    default_early_stopping_strategy: str = 'composite'
    
    # Logging settings
    log_level: str = 'INFO'
    enable_detailed_logging: bool = True
    
    # Performance settings
    enable_performance_tracking: bool = True
    max_validation_history: int = 1000


class ValidatorFactory:
    """Factory for creating validators."""
    
    @staticmethod
    def create_data_validator(
        step_name: str = "data_validation",
        required_fields: List[str] = None,
        data_types: Dict[str, Type] = None,
        value_ranges: Dict[str, tuple] = None,
        **kwargs
    ) -> DataValidator:
        """Create a data validator with common configuration."""
        config = {
            'required_fields': required_fields or ['features', 'targets'],
            'data_types': data_types or {'features': list, 'targets': list},
            'value_ranges': value_ranges or {},
            **kwargs
        }
        return DataValidator(step_name, config)
    
    @staticmethod
    def create_model_validator(
        step_name: str = "model_validation",
        required_methods: List[str] = None,
        performance_thresholds: Dict[str, float] = None,
        **kwargs
    ) -> ModelValidator:
        """Create a model validator with common configuration."""
        config = {
            'required_methods': required_methods or ['fit', 'predict'],
            'performance_thresholds': performance_thresholds or {},
            **kwargs
        }
        return ModelValidator(step_name, config)
    
    @staticmethod
    def create_config_validator(
        step_name: str = "config_validation",
        required_keys: List[str] = None,
        optional_keys: List[str] = None,
        value_validators: Dict[str, callable] = None,
        **kwargs
    ) -> ConfigValidator:
        """Create a config validator with common configuration."""
        config = {
            'required_keys': required_keys or ['model_type'],
            'optional_keys': optional_keys or [],
            'value_validators': value_validators or {},
            **kwargs
        }
        return ConfigValidator(step_name, config)
    
    @staticmethod
    def create_ml_validator_suite(
        data_config: Dict[str, Any] = None,
        model_config: Dict[str, Any] = None,
        config_config: Dict[str, Any] = None
    ) -> Dict[str, BaseValidator]:
        """Create a complete suite of validators for ML workflows."""
        validators = {}
        
        # Data validator
        if data_config is None:
            data_config = {
                'required_fields': ['features', 'targets'],
                'data_types': {'features': list, 'targets': list},
                'value_ranges': {'learning_rate': (0.001, 1.0)}
            }
        validators['data'] = ValidatorFactory.create_data_validator(**data_config)
        
        # Model validator
        if model_config is None:
            model_config = {
                'required_methods': ['fit', 'predict', 'score'],
                'performance_thresholds': {'accuracy': 0.8, 'f1_score': 0.7}
            }
        validators['model'] = ValidatorFactory.create_model_validator(**model_config)
        
        # Config validator
        if config_config is None:
            config_config = {
                'required_keys': ['model_type', 'hyperparameters'],
                'optional_keys': ['early_stopping', 'validation_split'],
                'value_validators': {
                    'learning_rate': lambda x: 0 < x < 1,
                    'batch_size': lambda x: x > 0 and isinstance(x, int)
                }
            }
        validators['config'] = ValidatorFactory.create_config_validator(**config_config)
        
        logger.info(f"Created ML validator suite with {len(validators)} validators")
        return validators


class EarlyStoppingFactory:
    """Factory for creating early stopping strategies."""
    
    @staticmethod
    def create_adaptive_patience_strategy(
        patience: int = 5,
        threshold: float = 0.001,
        adaptive: bool = True,
        min_patience: int = 3,
        max_patience: int = 20,
        direction: str = 'maximize'
    ) -> AdaptivePatienceStrategy:
        """Create an adaptive patience strategy."""
        config = EarlyStoppingConfig(
            early_stopping_patience=patience,
            early_stopping_threshold=threshold,
            adaptive_patience=adaptive,
            min_patience=min_patience,
            max_patience=max_patience,
            direction=direction
        )
        return AdaptivePatienceStrategy(config)
    
    @staticmethod
    def create_convergence_strategy(
        window: int = 10,
        min_improvement_rate: float = 0.001,
        direction: str = 'maximize'
    ) -> ConvergenceBasedStrategy:
        """Create a convergence-based strategy."""
        config = EarlyStoppingConfig(
            convergence_window=window,
            min_improvement_rate=min_improvement_rate,
            direction=direction
        )
        return ConvergenceBasedStrategy(config)
    
    @staticmethod
    def create_performance_strategy(
        threshold: float = 0.95,
        window: int = 5,
        direction: str = 'maximize'
    ) -> PerformanceBasedStrategy:
        """Create a performance-based strategy."""
        config = EarlyStoppingConfig(
            performance_threshold=threshold,
            performance_window=window,
            direction=direction
        )
        return PerformanceBasedStrategy(config)
    
    @staticmethod
    def create_time_strategy(
        max_time_seconds: int = 3600,
        direction: str = 'maximize'
    ) -> TimeBasedStrategy:
        """Create a time-based strategy."""
        config = EarlyStoppingConfig(
            max_time_seconds=max_time_seconds,
            direction=direction
        )
        return TimeBasedStrategy(config)
    
    @staticmethod
    def create_trial_strategy(
        max_trials: int = 1000,
        min_trials: int = 10,
        patience: int = 5,
        threshold: float = 0.001,
        direction: str = 'maximize'
    ) -> TrialBasedStrategy:
        """Create a trial-based strategy."""
        config = EarlyStoppingConfig(
            max_trials=max_trials,
            min_trials=min_trials,
            early_stopping_patience=patience,
            early_stopping_threshold=threshold,
            direction=direction
        )
        return TrialBasedStrategy(config)
    
    @staticmethod
    def create_composite_strategy(
        strategies: List[str] = None,
        config: EarlyStoppingConfig = None
    ) -> CompositeStrategy:
        """Create a composite strategy with multiple sub-strategies."""
        if strategies is None:
            strategies = ['adaptive', 'convergence', 'time', 'trial']
        
        if config is None:
            config = EarlyStoppingConfig()
        
        strategy_objects = []
        for strategy_name in strategies:
            if strategy_name == 'adaptive':
                strategy_objects.append(AdaptivePatienceStrategy(config))
            elif strategy_name == 'convergence':
                strategy_objects.append(ConvergenceBasedStrategy(config))
            elif strategy_name == 'performance':
                strategy_objects.append(PerformanceBasedStrategy(config))
            elif strategy_name == 'time':
                strategy_objects.append(TimeBasedStrategy(config))
            elif strategy_name == 'trial':
                strategy_objects.append(TrialBasedStrategy(config))
            else:
                logger.warning(f"Unknown strategy: {strategy_name}")
        
        return CompositeStrategy(strategy_objects, config)
    
    @staticmethod
    def create_strategy_suite(
        include_adaptive: bool = True,
        include_convergence: bool = True,
        include_performance: bool = True,
        include_time: bool = True,
        include_trial: bool = True,
        config: EarlyStoppingConfig = None
    ) -> Dict[str, EarlyStoppingStrategy]:
        """Create a complete suite of early stopping strategies."""
        strategies = {}
        
        if config is None:
            config = EarlyStoppingConfig()
        
        if include_adaptive:
            strategies['adaptive'] = AdaptivePatienceStrategy(config)
        
        if include_convergence:
            strategies['convergence'] = ConvergenceBasedStrategy(config)
        
        if include_performance:
            strategies['performance'] = PerformanceBasedStrategy(config)
        
        if include_time:
            strategies['time'] = TimeBasedStrategy(config)
        
        if include_trial:
            strategies['trial'] = TrialBasedStrategy(config)
        
        # Always include composite
        strategy_list = list(strategies.keys())
        strategies['composite'] = EarlyStoppingFactory.create_composite_strategy(strategy_list, config)
        
        logger.info(f"Created early stopping strategy suite with {len(strategies)} strategies")
        return strategies


class ProductionMLFactory:
    """Main factory for creating production-ready ML components."""
    
    def __init__(self, config: ProductionConfig = None):
        """Initialize the production factory."""
        self.config = config or ProductionConfig()
        self.validators = {}
        self.early_stopping_strategies = {}
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level.upper()))
        
        # Initialize components if enabled
        if self.config.enable_data_validation or self.config.enable_model_validation or self.config.enable_config_validation:
            self._setup_validators()
        
        if self.config.enable_early_stopping:
            self._setup_early_stopping()
        
        logger.info("🏭 Production ML Factory initialized")
    
    def _setup_validators(self):
        """Setup validators based on configuration."""
        if self.config.enable_data_validation:
            self.validators['data'] = ValidatorFactory.create_data_validator()
        
        if self.config.enable_model_validation:
            self.validators['model'] = ValidatorFactory.create_model_validator()
        
        if self.config.enable_config_validation:
            self.validators['config'] = ValidatorFactory.create_config_validator()
        
        logger.info(f"Setup {len(self.validators)} validators")
    
    def _setup_early_stopping(self):
        """Setup early stopping strategies based on configuration."""
        self.early_stopping_strategies = EarlyStoppingFactory.create_strategy_suite()
        logger.info(f"Setup {len(self.early_stopping_strategies)} early stopping strategies")
    
    def get_validator(self, validator_type: str) -> Optional[BaseValidator]:
        """Get a specific validator."""
        return self.validators.get(validator_type)
    
    def get_early_stopping_strategy(self, strategy_name: str) -> Optional[EarlyStoppingStrategy]:
        """Get a specific early stopping strategy."""
        return self.early_stopping_strategies.get(strategy_name)
    
    def get_default_early_stopping_strategy(self) -> Optional[EarlyStoppingStrategy]:
        """Get the default early stopping strategy."""
        return self.early_stopping_strategies.get(self.config.default_early_stopping_strategy)
    
    def create_custom_validator(
        self,
        validator_type: Type[BaseValidator],
        step_name: str,
        config: Dict[str, Any]
    ) -> BaseValidator:
        """Create a custom validator."""
        validator = validator_type(step_name, config)
        self.validators[step_name] = validator
        logger.info(f"Created custom validator: {step_name}")
        return validator
    
    def create_custom_early_stopping_strategy(
        self,
        strategy_type: Type[EarlyStoppingStrategy],
        strategy_name: str,
        config: EarlyStoppingConfig = None
    ) -> EarlyStoppingStrategy:
        """Create a custom early stopping strategy."""
        strategy = strategy_type(config)
        self.early_stopping_strategies[strategy_name] = strategy
        logger.info(f"Created custom early stopping strategy: {strategy_name}")
        return strategy
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get a comprehensive system summary."""
        summary = {
            'config': {
                'enable_data_validation': self.config.enable_data_validation,
                'enable_model_validation': self.config.enable_model_validation,
                'enable_config_validation': self.config.enable_config_validation,
                'enable_early_stopping': self.config.enable_early_stopping,
                'default_early_stopping_strategy': self.config.default_early_stopping_strategy
            },
            'validators': {
                name: validator.get_validation_summary() 
                for name, validator in self.validators.items()
            },
            'early_stopping_strategies': {
                name: {
                    'type': strategy.__class__.__name__,
                    'stopping_reason': strategy.get_stopping_reason()
                }
                for name, strategy in self.early_stopping_strategies.items()
            },
            'total_components': len(self.validators) + len(self.early_stopping_strategies)
        }
        
        return summary


# Convenience functions for quick setup
def create_production_system(config: ProductionConfig = None) -> ProductionMLFactory:
    """Create a production ML system with all components."""
    return ProductionMLFactory(config)


def create_ml_validator_suite(**kwargs) -> Dict[str, BaseValidator]:
    """Create a complete ML validator suite."""
    return ValidatorFactory.create_ml_validator_suite(**kwargs)


def create_early_stopping_suite(**kwargs) -> Dict[str, EarlyStoppingStrategy]:
    """Create a complete early stopping strategy suite."""
    return EarlyStoppingFactory.create_strategy_suite(**kwargs)


def create_quick_validation_system() -> ProductionMLFactory:
    """Create a quick validation system with minimal configuration."""
    config = ProductionConfig(
        enable_detailed_logging=False,
        log_level='WARNING'
    )
    return ProductionMLFactory(config)


def create_full_validation_system() -> ProductionMLFactory:
    """Create a full validation system with all features enabled."""
    config = ProductionConfig(
        enable_data_validation=True,
        enable_model_validation=True,
        enable_config_validation=True,
        enable_early_stopping=True,
        enable_performance_tracking=True,
        enable_detailed_logging=True,
        log_level='DEBUG'
    )
    return ProductionMLFactory(config)


# Example usage and testing
if __name__ == "__main__":
    # Create a production system
    system = create_production_system()
    
    # Get system summary
    summary = system.get_system_summary()
    print("Production System Summary:")
    print(f"Total components: {summary['total_components']}")
    print(f"Validators: {list(system.validators.keys())}")
    print(f"Early stopping strategies: {list(system.early_stopping_strategies.keys())}")
    
    # Test validators
    data_validator = system.get_validator('data')
    if data_validator:
        test_data = {'features': [[1, 2, 3], [4, 5, 6]], 'targets': [0, 1]}
        is_valid = data_validator.is_valid(test_data)
        print(f"Data validation test: {'PASSED' if is_valid else 'FAILED'}")
    
    # Test early stopping
    early_stopping = system.get_default_early_stopping_strategy()
    if early_stopping:
        test_history = [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.83, 0.84, 0.845, 0.847]
        should_stop = early_stopping.should_stop(test_history, 10)
        print(f"Early stopping test: {'STOP' if should_stop else 'CONTINUE'}")
    
    print("✅ Production factory test completed successfully!")