"""
Integrated Example: Combining All Architectural Improvements

This module demonstrates how to use all the architectural improvements together:
- Dependency Injection
- Standard Interfaces
- Configuration-Driven Design
- Modular Components with Single Responsibility
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path

import yaml

from .config_driven_architecture import ConfigLoader

# Import our architectural components
from .dependency_injection import DIContainer, inject
from .modular_components import (
    DataQualityValidator,
    ExchangeDataSourceFactory,
    IDataSource,
    IDataValidator,
    IFeatureCalculator,
    IModelTrainer,
    LocalDataSource,
    ModelTrainerFactory,
    PriceFeatureCalculator,
    SchemaValidator,
    VolumeFeatureCalculator,
)
from .standard_interfaces import BasePipelineStep, IPipelineStep, StepConfig, StepResult

# ============================================================================
# Step Implementations using Standard Interfaces
# ============================================================================

@inject(data_source='data_source', logger='logger')
class DataLoadingStep(BasePipelineStep):
    """Data loading step with dependency injection."""
    
    def __init__(self, config: StepConfig, data_source: IDataSource = None, logger=None):
        super().__init__(config, logger)
        self.data_source = data_source
    
    @property
    def version(self) -> str:
        return "2.0.0"
    
    async def validate_inputs(self, **kwargs) -> bool:
        """Validate input parameters."""
        required = ['symbol', 'start_date', 'end_date']
        for param in required:
            if param not in kwargs:
                self.add_warning(f"Missing required parameter: {param}")
                return False
        return True
    
    async def _execute_impl(self, **kwargs) -> pd.DataFrame:
        """Load data using injected data source."""
        symbol = kwargs['symbol']
        start = datetime.fromisoformat(kwargs['start_date'])
        end = datetime.fromisoformat(kwargs['end_date'])
        
        # Use injected data source
        data = await self.data_source.fetch_data(symbol, start, end)
        
        # Add metrics
        self.add_metric('rows_loaded', len(data))
        self.add_metric('date_range', f"{start.date()} to {end.date()}")
        
        return data


@inject(validators='validators', logger='logger')
class ValidationStep(BasePipelineStep):
    """Data validation step with multiple validators."""
    
    def __init__(self, config: StepConfig, validators: list[IDataValidator] = None, logger=None):
        super().__init__(config, logger)
        self.validators = validators or []
    
    async def _execute_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data using injected validators."""
        all_valid = True
        
        for validator in self.validators:
            result = validator.validate(data)
            
            if not result.is_valid:
                for error in result.errors:
                    self.add_warning(f"{validator.__class__.__name__}: {error}")
                all_valid = False
            
            # Add validation metrics
            for metric_name, value in result.metrics.items():
                self.add_metric(f"{validator.__class__.__name__}_{metric_name}", value)
        
        if not all_valid:
            raise ValueError("Data validation failed")
        
        return data


@inject(calculators='feature_calculators', logger='logger')
class FeatureEngineeringStep(BasePipelineStep):
    """Feature engineering with modular calculators."""
    
    def __init__(self, config: StepConfig, calculators: list[IFeatureCalculator] = None, logger=None):
        super().__init__(config, logger)
        self.calculators = calculators or []
    
    async def _execute_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features using injected calculators."""
        import pandas as pd
        
        features = pd.DataFrame(index=data.index)
        
        for calculator in self.calculators:
            calc_features = calculator.calculate(data)
            features = pd.concat([features, calc_features], axis=1)
            
            # Track which features came from which calculator
            self.add_metric(
                f"{calculator.__class__.__name__}_features",
                calculator.get_feature_names()
            )
        
        self.add_metric('total_features', len(features.columns))
        return features


@inject(trainer='model_trainer', logger='logger')
class ModelTrainingStep(BasePipelineStep):
    """Model training with injected trainer."""
    
    def __init__(self, config: StepConfig, trainer: IModelTrainer = None, logger=None):
        super().__init__(config, logger)
        self.trainer = trainer
    
    async def _execute_impl(self, features: pd.DataFrame, labels: pd.Series = None) -> Any:
        """Train model using injected trainer."""
        # Create simple labels if not provided
        if labels is None:
            labels = (features.iloc[:, 0] > features.iloc[:, 0].shift(1)).astype(int)
        
        # Fill NaN values
        features_clean = features.fillna(0)
        
        # Train model
        model = self.trainer.train(features_clean, labels)
        
        # Add training metrics
        self.add_metric('training_samples', len(features))
        self.add_metric('hyperparameters', self.trainer.get_hyperparameters())
        
        return model


# ============================================================================
# Configuration-Driven Pipeline with Dependency Injection
# ============================================================================

class IntegratedPipeline:
    """
    Main pipeline that integrates all architectural improvements.
    """
    
    def __init__(self, config_path: str):
        # Load configuration
        self.config = ConfigLoader.load_from_file(config_path)
        
        # Setup dependency injection container
        self.container = self._setup_container()
        
        # Initialize pipeline steps
        self.steps = self._initialize_steps()
        
        # Setup logging
        self.logger = self.container.get('logger')
    
    def _setup_container(self) -> DIContainer:
        """Setup DI container based on configuration."""
        container = DIContainer()
        
        # Register logger
        container.register(
            'logger',
            logging.Logger,
            lambda: self._create_logger(),
            singleton=True
        )
        
        # Register data source based on config
        data_source_config = self.config.global_settings.get('data_source', {})
        source_type = data_source_config.get('type', 'local')
        
        if source_type == 'local':
            container.register(
                'data_source',
                IDataSource,
                lambda: LocalDataSource(data_source_config.get('data_dir', 'data')),
                singleton=True
            )
        else:
            # Use factory for exchange data sources
            exchange = data_source_config.get('exchange', source_type)
            container.register(
                'data_source',
                IDataSource,
                lambda: ExchangeDataSourceFactory.create(
                    exchange,
                    api_key=data_source_config.get('api_key'),
                    api_secret=data_source_config.get('api_secret'),
                    testnet=data_source_config.get('testnet', False)
                ),
                singleton=True
            )
        
        # Register validators
        container.register_instance(
            'validators',
            [
                SchemaValidator(['open', 'high', 'low', 'close', 'volume']),
                DataQualityValidator(max_null_percentage=0.05),
            ]
        )
        
        # Register feature calculators
        container.register_instance(
            'feature_calculators',
            [
                PriceFeatureCalculator(),
                VolumeFeatureCalculator(window=20),
            ]
        )
        
        # Register model trainer using factory
        trainer_config = self.config.global_settings.get('model', {})
        model_type = trainer_config.get('type', 'lightgbm')
        
        container.register(
            'model_trainer',
            IModelTrainer,
            lambda: ModelTrainerFactory.create(
                model_type,
                **trainer_config.get('hyperparameters', {})
            ),
            singleton=False
        )
        
        return container
    
    def _create_logger(self) -> logging.Logger:
        """Create configured logger."""
        logger = logging.getLogger('IntegratedPipeline')
        logger.setLevel(
            getattr(logging, self.config.global_settings.get('log_level', 'INFO'))
        )
        
        # Add console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _initialize_steps(self) -> dict[str, IPipelineStep]:
        """Initialize pipeline steps from configuration."""
        steps = {}
        
        # Map step types to classes
        step_classes = {
            'data_loading': DataLoadingStep,
            'validation': ValidationStep,
            'feature_engineering': FeatureEngineeringStep,
            'model_training': ModelTrainingStep,
        }
        
        # Create steps from config
        for step_name, step_config in self.config.steps.items():
            if not step_config.enabled:
                continue
            
            # Convert to StepConfig
            config = StepConfig(
                name=step_name,
                enabled=step_config.enabled,
                parameters=step_config.parameters
            )
            
            # Get step class
            step_type = step_config.parameters.get('type', step_name)
            step_class = step_classes.get(step_type)
            
            if step_class:
                # Create step with DI container
                step = step_class(config, _container=self.container)
                steps[step_name] = step
            else:
                self.logger.warning(f"Unknown step type: {step_type}")
        
        return steps
    
    async def run(self) -> dict[str, StepResult]:
        """Execute the pipeline."""
        results = {}
        step_outputs = {}
        
        # Sort steps by order
        sorted_steps = sorted(
            [(name, self.config.steps[name]) for name in self.steps.keys()],
            key=lambda x: x[1].order
        )
        
        for step_name, step_config in sorted_steps:
            self.logger.info(f"Executing step: {step_name}")
            
            step = self.steps[step_name]
            
            # Prepare inputs from configuration
            inputs = {}
            
            # Add parameters from config
            inputs.update(step_config.parameters)
            
            # Add outputs from previous steps
            for input_name, input_ref in step_config.inputs.items():
                if '.' in input_ref:
                    ref_step, ref_output = input_ref.split('.')
                    if ref_step in step_outputs:
                        inputs[input_name] = step_outputs[ref_step]
                
            # Execute step
            result = await step.execute(**inputs)
            results[step_name] = result
            
            if result.is_success:
                step_outputs[step_name] = result.data
                self.logger.info(
                    f"Step {step_name} completed in {result.duration:.2f}s"
                )
            else:
                self.logger.error(f"Step {step_name} failed: {result.error}")
                if self.config.global_settings.get('fail_fast', True):
                    break
        
        return results


# ============================================================================
# Example Usage
# ============================================================================

async def create_example_config():
    """Create an example configuration file."""
    
    config = {
        "name": "Integrated_ML_Pipeline",
        "version": "2.0.0",
        "description": "Example pipeline with all architectural improvements",
        
        "global_settings": {
            "log_level": "INFO",
            "fail_fast": True,
            "data_source": {
                "type": "local",
                "data_dir": "data/cache"
            },
            "model": {
                "hyperparameters": {
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "n_estimators": 100
                }
            }
        },
        
        "steps": {
            "data_loading": {
                "class_name": "DataLoadingStep",
                "enabled": True,
                "order": 1,
                "parameters": {
                    "type": "data_loading",
                    "symbol": "BTCUSDT",
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31"
                },
                "outputs": ["data"]
            },
            
            "validation": {
                "class_name": "ValidationStep",
                "enabled": True,
                "order": 2,
                "inputs": {
                    "data": "data_loading.data"
                },
                "parameters": {
                    "type": "validation"
                },
                "outputs": ["validated_data"]
            },
            
            "feature_engineering": {
                "class_name": "FeatureEngineeringStep",
                "enabled": True,
                "order": 3,
                "inputs": {
                    "data": "validation.validated_data"
                },
                "parameters": {
                    "type": "feature_engineering"
                },
                "outputs": ["features"]
            },
            
            "model_training": {
                "class_name": "ModelTrainingStep",
                "enabled": True,
                "order": 4,
                "inputs": {
                    "features": "feature_engineering.features"
                },
                "parameters": {
                    "type": "model_training"
                },
                "outputs": ["model"]
            }
        }
    }
    
    # Save configuration
    config_path = Path("config/integrated_pipeline.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


async def main():
    """Main entry point demonstrating the integrated architecture."""
    
    # Create example configuration
    config_path = await create_example_config()
    print(f"Created configuration at: {config_path}")
    
    # Create and run pipeline
    pipeline = IntegratedPipeline(str(config_path))
    
    print("\nRunning integrated pipeline...")
    results = await pipeline.run()
    
    # Print results
    print("\nPipeline Results:")
    for step_name, result in results.items():
        print(f"\n{step_name}:")
        print(f"  Status: {result.status.value}")
        print(f"  Duration: {result.duration:.2f}s" if result.duration else "  Duration: N/A")
        print(f"  Metrics: {result.metrics}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")
        if result.error:
            print(f"  Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())