"""
Integration Example

This module demonstrates how to use all the implemented abstract base classes
and concrete implementations together in a production-ready system.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
import logging

# Import our implemented classes
from src.utils.base_validator import (
    BaseValidator, DataValidator, ModelValidator, ConfigValidator
)
from src.utils.standalone_early_stopping import (
    EarlyStoppingStrategy, AdaptivePatienceStrategy, ConvergenceBasedStrategy,
    PerformanceBasedStrategy, TimeBasedStrategy, TrialBasedStrategy, CompositeStrategy,
    create_default_strategy, EarlyStoppingConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionMLSystem:
    """
    Production ML System demonstrating integration of all implemented components.
    """
    
    def __init__(self):
        """Initialize the production ML system."""
        self.validators = {}
        self.early_stopping_strategies = {}
        self.training_history = []
        self.validation_history = []
        
        # Initialize validators
        self._setup_validators()
        
        # Initialize early stopping strategies
        self._setup_early_stopping()
        
        logger.info("🚀 Production ML System initialized")
    
    def _setup_validators(self):
        """Setup all validators for the system."""
        # Data validator
        data_config = {
            'required_fields': ['features', 'targets'],
            'data_types': {'features': list, 'targets': list},
            'value_ranges': {'learning_rate': (0.001, 1.0)}
        }
        self.validators['data'] = DataValidator('data_validation', data_config)
        
        # Model validator
        model_config = {
            'required_methods': ['fit', 'predict', 'score'],
            'performance_thresholds': {'accuracy': 0.8, 'f1_score': 0.7}
        }
        self.validators['model'] = ModelValidator('model_validation', model_config)
        
        # Config validator
        config_config = {
            'required_keys': ['model_type', 'hyperparameters'],
            'optional_keys': ['early_stopping', 'validation_split'],
            'value_validators': {
                'learning_rate': lambda x: 0 < x < 1,
                'batch_size': lambda x: x > 0 and isinstance(x, int)
            }
        }
        self.validators['config'] = ConfigValidator('config_validation', config_config)
        
        logger.info("✅ Validators setup complete")
    
    def _setup_early_stopping(self):
        """Setup early stopping strategies."""
        # Create different strategies for different scenarios
        self.early_stopping_strategies['adaptive'] = AdaptivePatienceStrategy()
        self.early_stopping_strategies['convergence'] = ConvergenceBasedStrategy()
        self.early_stopping_strategies['performance'] = PerformanceBasedStrategy()
        self.early_stopping_strategies['time'] = TimeBasedStrategy()
        self.early_stopping_strategies['trial'] = TrialBasedStrategy()
        self.early_stopping_strategies['composite'] = create_default_strategy()
        
        logger.info("✅ Early stopping strategies setup complete")
    
    async def validate_training_data(self, data: Dict[str, Any]) -> bool:
        """Validate training data using data validator."""
        logger.info("🔍 Validating training data...")
        
        try:
            result = await self.validators['data'].validate(data)
            self.validation_history.append({
                'type': 'data_validation',
                'result': result,
                'timestamp': time.time()
            })
            
            if result['success']:
                logger.info("✅ Training data validation passed")
                return True
            else:
                logger.error(f"❌ Training data validation failed: {result['errors']}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Data validation error: {e}")
            return False
    
    async def validate_model_config(self, config: Dict[str, Any]) -> bool:
        """Validate model configuration using config validator."""
        logger.info("🔍 Validating model configuration...")
        
        try:
            result = await self.validators['config'].validate(config)
            self.validation_history.append({
                'type': 'config_validation',
                'result': result,
                'timestamp': time.time()
            })
            
            if result['success']:
                logger.info("✅ Model configuration validation passed")
                return True
            else:
                logger.error(f"❌ Model configuration validation failed: {result['errors']}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Config validation error: {e}")
            return False
    
    async def validate_trained_model(self, model: Any, performance_metrics: Dict[str, float]) -> bool:
        """Validate trained model using model validator."""
        logger.info("🔍 Validating trained model...")
        
        try:
            context = {'performance_metrics': performance_metrics}
            result = await self.validators['model'].validate(model, context)
            self.validation_history.append({
                'type': 'model_validation',
                'result': result,
                'timestamp': time.time()
            })
            
            if result['success']:
                logger.info("✅ Model validation passed")
                return True
            else:
                logger.error(f"❌ Model validation failed: {result['errors']}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model validation error: {e}")
            return False
    
    def should_stop_training(self, strategy_name: str, training_history: List[float], current_epoch: int) -> bool:
        """Check if training should stop using specified strategy."""
        if strategy_name not in self.early_stopping_strategies:
            logger.warning(f"⚠️ Unknown strategy: {strategy_name}, using composite")
            strategy_name = 'composite'
        
        strategy = self.early_stopping_strategies[strategy_name]
        should_stop = strategy.should_stop(training_history, current_epoch)
        
        if should_stop:
            reason = strategy.get_stopping_reason()
            logger.info(f"⏹️ Early stopping triggered: {reason}")
        
        return should_stop
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        summary = {
            'total_validations': len(self.validation_history),
            'validation_types': {},
            'success_rate': 0.0,
            'validators': {}
        }
        
        # Count by type
        for validation in self.validation_history:
            vtype = validation['type']
            if vtype not in summary['validation_types']:
                summary['validation_types'][vtype] = {'total': 0, 'successful': 0}
            
            summary['validation_types'][vtype]['total'] += 1
            if validation['result']['success']:
                summary['validation_types'][vtype]['successful'] += 1
        
        # Calculate success rates
        for vtype, counts in summary['validation_types'].items():
            counts['success_rate'] = counts['successful'] / counts['total'] if counts['total'] > 0 else 0.0
        
        # Overall success rate
        total_successful = sum(v['successful'] for v in summary['validation_types'].values())
        summary['success_rate'] = total_successful / summary['total_validations'] if summary['total_validations'] > 0 else 0.0
        
        # Validator summaries
        for name, validator in self.validators.items():
            summary['validators'][name] = validator.get_validation_summary()
        
        return summary
    
    def get_early_stopping_summary(self) -> Dict[str, Any]:
        """Get early stopping strategies summary."""
        summary = {
            'strategies': {},
            'total_strategies': len(self.early_stopping_strategies)
        }
        
        for name, strategy in self.early_stopping_strategies.items():
            summary['strategies'][name] = {
                'type': strategy.__class__.__name__,
                'stopping_reason': strategy.get_stopping_reason(),
                'trials_without_improvement': getattr(strategy, 'trials_without_improvement', 0),
                'best_value': getattr(strategy, 'best_value', None),
                'best_trial': getattr(strategy, 'best_trial', 0)
            }
        
        return summary


class MockModel:
    """Mock model for demonstration purposes."""
    
    def __init__(self, model_type: str = "mock"):
        self.model_type = model_type
        self.is_fitted = False
        self.training_history = []
    
    def fit(self, X, y):
        """Mock fit method."""
        self.is_fitted = True
        self.training_history = [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.83, 0.84, 0.845, 0.847]
        return self
    
    def predict(self, X):
        """Mock predict method."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return [0.1, 0.2, 0.3]  # Mock predictions
    
    def score(self, X, y):
        """Mock score method."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return 0.85  # Mock score


async def demonstrate_integration():
    """Demonstrate the integration of all implemented components."""
    logger.info("🎯 Starting integration demonstration...")
    
    # Initialize the production system
    system = ProductionMLSystem()
    
    # Prepare mock data
    training_data = {
        'features': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        'targets': [0, 1, 0],
        'learning_rate': 0.01
    }
    
    model_config = {
        'model_type': 'mock_model',
        'hyperparameters': {
            'learning_rate': 0.01,
            'batch_size': 32
        },
        'early_stopping': True,
        'validation_split': 0.2
    }
    
    # Step 1: Validate training data
    logger.info("\n📊 Step 1: Validating training data")
    data_valid = await system.validate_training_data(training_data)
    
    # Step 2: Validate model configuration
    logger.info("\n⚙️ Step 2: Validating model configuration")
    config_valid = await system.validate_model_config(model_config)
    
    # Step 3: Create and train mock model
    logger.info("\n🤖 Step 3: Training mock model")
    model = MockModel()
    model.fit(training_data['features'], training_data['targets'])
    
    # Step 4: Validate trained model
    logger.info("\n✅ Step 4: Validating trained model")
    performance_metrics = {'accuracy': 0.85, 'f1_score': 0.82}
    model_valid = await system.validate_trained_model(model, performance_metrics)
    
    # Step 5: Test early stopping strategies
    logger.info("\n⏹️ Step 5: Testing early stopping strategies")
    training_history = [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.83, 0.84, 0.845, 0.847]
    
    for strategy_name in ['adaptive', 'convergence', 'performance', 'time', 'trial', 'composite']:
        should_stop = system.should_stop_training(strategy_name, training_history, 10)
        logger.info(f"   {strategy_name}: should_stop = {should_stop}")
    
    # Step 6: Get comprehensive summaries
    logger.info("\n📈 Step 6: Getting system summaries")
    
    validation_summary = system.get_validation_summary()
    logger.info(f"Validation Summary: {validation_summary}")
    
    early_stopping_summary = system.get_early_stopping_summary()
    logger.info(f"Early Stopping Summary: {early_stopping_summary}")
    
    # Step 7: Demonstrate async validation
    logger.info("\n🔄 Step 7: Demonstrating async validation")
    
    # Test multiple validations concurrently
    validation_tasks = [
        system.validate_training_data(training_data),
        system.validate_model_config(model_config),
        system.validate_trained_model(model, performance_metrics)
    ]
    
    results = await asyncio.gather(*validation_tasks, return_exceptions=True)
    logger.info(f"Concurrent validation results: {results}")
    
    logger.info("\n🎉 Integration demonstration completed successfully!")


def demonstrate_synchronous_usage():
    """Demonstrate synchronous usage of the validators."""
    logger.info("🔄 Demonstrating synchronous usage...")
    
    # Create validators
    data_validator = DataValidator('sync_data', {
        'required_fields': ['features'],
        'data_types': {'features': list}
    })
    
    # Test synchronous validation
    test_data = {'features': [[1, 2, 3], [4, 5, 6]]}
    is_valid = data_validator.is_valid(test_data)
    logger.info(f"Synchronous validation result: {is_valid}")
    
    # Get validation summary
    summary = data_validator.get_validation_summary()
    logger.info(f"Validator summary: {summary}")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_integration())
    
    # Demonstrate synchronous usage
    demonstrate_synchronous_usage()