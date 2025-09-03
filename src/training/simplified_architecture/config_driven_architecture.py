from typing import Dict, List, Optional, Union, Any, Tuple
"""
Configuration-Driven Architecture for ML Pipeline

This module implements a configuration-driven approach where complexity
is moved from code to configuration files, making the system more flexible
and easier to understand.
"""
import importlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union
import yaml
import asyncio

class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = 'json'
    YAML = 'yaml'
    PYTHON = 'py'

@dataclass
class PipelineConfig:
    """Main pipeline configuration structure."""
    name: str
    version: str
    description: str = ''
    steps: Dict[str, 'StepConfig'] = None
    global_settings: Dict[str, Any] = None
    dependencies: Dict[str, 'DependencyConfig'] = None

    def __post_init__(self) -> None:
        if self.steps is None:
            self.steps = {}
        if self.global_settings is None:
            self.global_settings = {}
        if self.dependencies is None:
            self.dependencies = {}

@dataclass
class StepConfig:
    """Configuration for a single pipeline step."""
    class_name: str
    enabled: bool = True
    order: int = 0
    parameters: Dict[str, Any] = None
    inputs: Dict[str, str] = None
    outputs: list[str] = None
    retry_policy: 'RetryPolicy' = None
    validation: 'ValidationConfig' = None

    def __post_init__(self) -> None:
        if self.parameters is None:
            self.parameters = {}
        if self.inputs is None:
            self.inputs = {}
        if self.outputs is None:
            self.outputs = []

@dataclass
class DependencyConfig:
    """Configuration for a dependency."""
    type: str
    class_name: str
    module: str
    parameters: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.parameters is None:
            self.parameters = {}

@dataclass
class RetryPolicy:
    """Retry configuration for steps."""
    max_attempts: int = 3
    backoff_type: str = 'exponential'
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    retry_on_exceptions: list[str] = None

    def __post_init__(self) -> None:
        if self.retry_on_exceptions is None:
            self.retry_on_exceptions = ['Exception']

@dataclass
class ValidationConfig:
    """Validation configuration for steps."""
    input_schema: Dict[str, Any] = None
    output_schema: Dict[str, Any] = None
    custom_validators: list[str] = None
    fail_on_warning: bool = False

    def __post_init__(self) -> None:
        if self.input_schema is None:
            self.input_schema = {}
        if self.output_schema is None:
            self.output_schema = {}
        if self.custom_validators is None:
            self.custom_validators = []

class ConfigLoader:
    """Loads and validates configuration from various sources."""

    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> PipelineConfig:
        """Load configuration from a file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f'Configuration file not found: {file_path}')
        format_map = {'.json': ConfigFormat.JSON, '.yaml': ConfigFormat.YAML, '.yml': ConfigFormat.YAML, '.py': ConfigFormat.PYTHON}
        file_format = format_map.get(file_path.suffix.lower())
        if not file_format:
            raise ValueError(f'Unsupported configuration format: {file_path.suffix}')
        if file_format == ConfigFormat.JSON:
            return ConfigLoader._load_json(file_path)
        elif file_format == ConfigFormat.YAML:
            return ConfigLoader._load_yaml(file_path)
        elif file_format == ConfigFormat.PYTHON:
            return ConfigLoader._load_python(file_path)

    @staticmethod
    def _load_json(file_path: Path) -> PipelineConfig:
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return ConfigLoader._parse_config_dict(data)

    @staticmethod
    def _load_yaml(file_path: Path) -> PipelineConfig:
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return ConfigLoader._parse_config_dict(data)

    @staticmethod
    def _load_python(file_path: Path) -> PipelineConfig:
        """Load configuration from Python file."""
        spec = importlib.util.spec_from_file_location('config', file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'PIPELINE_CONFIG'):
            return ConfigLoader._parse_config_dict(module.PIPELINE_CONFIG)
        else:
            raise ValueError(f'No PIPELINE_CONFIG found in {file_path}')

    @staticmethod
    def _parse_config_dict(data: Dict[str, Any]) -> PipelineConfig:
        """Parse configuration dictionary into dataclasses."""
        steps = {}
        for step_name, step_data in data.get('steps', {}).items():
            retry_data = step_data.get('retry_policy')
            retry_policy = RetryPolicy(**retry_data) if retry_data else None
            validation_data = step_data.get('validation')
            validation = ValidationConfig(**validation_data) if validation_data else None
            step_config = StepConfig(class_name=step_data['class_name'], enabled=step_data.get('enabled', True), order=step_data.get('order', 0), parameters=step_data.get('parameters', {}), inputs=step_data.get('inputs', {}), outputs=step_data.get('outputs', []), retry_policy=retry_policy, validation=validation)
            steps[step_name] = step_config
        dependencies = {}
        for dep_name, dep_data in data.get('dependencies', {}).items():
            dep_config = DependencyConfig(**dep_data)
            dependencies[dep_name] = dep_config
        return PipelineConfig(name=data['name'], version=data['version'], description=data.get('description', ''), steps=steps, global_settings=data.get('global_settings', {}), dependencies=dependencies)

class ConfigBuilder:
    """Builder pattern for creating configurations programmatically."""

    def __init__(self, name: str, version: str) -> None:
        self.config = PipelineConfig(name=name, version=version)

    def with_description(self, description: str) -> 'ConfigBuilder':
        """Add description to pipeline."""
        self.config.description = description
        return self

    def add_step(self, name: str, class_name: str, **kwargs) -> 'ConfigBuilder':
        """Add a step to the pipeline."""
        step_config = StepConfig(class_name=class_name, **kwargs)
        self.config.steps[name] = step_config
        return self

    def add_dependency(self, name: str, class_name: str, module: str, type: str='singleton', **kwargs) -> 'ConfigBuilder':
        """Add a dependency."""
        dep_config = DependencyConfig(type=type, class_name=class_name, module=module, parameters=kwargs)
        self.config.dependencies[name] = dep_config
        return self

    def with_global_setting(self, key: str, value: Any) -> 'ConfigBuilder':
        """Add a global setting."""
        self.config.global_settings[key] = value
        return self

    def build(self) -> PipelineConfig:
        """Build and return the configuration."""
        return self.config

    def save(self, file_path: Union[str, Path], format: ConfigFormat=ConfigFormat.YAML) -> None:
        """Save configuration to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = self._to_dict(self.config)
        if format == ConfigFormat.JSON:
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif format == ConfigFormat.YAML:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)

    def _to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert dataclass to dictionary recursively."""
        if hasattr(obj, '__dataclass_fields__'):
            return {k: self._to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_dict(v) for v in obj]
        else:
            return obj
EXAMPLE_YAML_CONFIG = '\nname: ML_Trading_Pipeline\nversion: 1.0.0\ndescription: Simplified ML pipeline for trading\n\nglobal_settings:\n  data_dir: data/cache\n  model_dir: models\n  log_level: INFO\n  max_parallel_steps: 3\n\ndependencies:\n  logger:\n    type: singleton\n    class_name: FileLogger\n    module: src.utils.logging\n    parameters:\n      log_file: logs/pipeline.log\n      \n  validator:\n    type: singleton\n    class_name: DataValidator\n    module: src.validation\n    \n  cache:\n    type: singleton\n    class_name: RedisCache\n    module: src.caching\n    parameters:\n      host: localhost\n      port: 6379\n\nsteps:\n  data_loading:\n    class_name: DataLoader\n    order: 1\n    enabled: true\n    parameters:\n      source: binance\n      symbol: BTCUSDT\n      timeframe: 1h\n      start_date: "2023-01-01"\n      end_date: "2023-12-31"\n    outputs:\n      - raw_data\n    validation:\n      output_schema:\n        type: dataframe\n        required_columns: [open, high, low, close, volume]\n        min_rows: 1000\n    \n  labeling:\n    class_name: TripleBarrierLabeler\n    order: 2\n    enabled: true\n    inputs:\n      data: data_loading.raw_data\n    parameters:\n      profit_target: 0.02\n      stop_loss: 0.01\n      time_barrier: 24\n    outputs:\n      - labeled_data\n    retry_policy:\n      max_attempts: 3\n      backoff_type: exponential\n      initial_delay_seconds: 1.0\n    \n  feature_engineering:\n    class_name: TechnicalFeatureExtractor\n    order: 3\n    enabled: true\n    inputs:\n      data: labeling.labeled_data\n    parameters:\n      indicators:\n        - name: RSI\n          periods: [14, 21]\n        - name: MACD\n          fast: 12\n          slow: 26\n          signal: 9\n        - name: BB\n          period: 20\n          std: 2\n    outputs:\n      - features\n      \n  model_training:\n    class_name: LightGBMTrainer\n    order: 4\n    enabled: true\n    inputs:\n      features: feature_engineering.features\n      labels: labeling.labeled_data\n    parameters:\n      model_params:\n        num_leaves: 31\n        learning_rate: 0.05\n        n_estimators: 100\n        objective: binary\n      validation_split: 0.2\n    outputs:\n      - trained_model\n      - training_metrics\n      \n  validation:\n    class_name: BacktestValidator\n    order: 5\n    enabled: true\n    inputs:\n      model: model_training.trained_model\n      data: feature_engineering.features\n    parameters:\n      metrics: [accuracy, precision, recall, sharpe_ratio]\n      test_period_days: 30\n    outputs:\n      - validation_report\n'

class ConfigDrivenPipeline:
    """Executes pipeline based on configuration."""

    def __init__(self, config: PipelineConfig, container: Any=None) -> None:
        self.config = config
        self.container = container
        self.steps = {}
        self.results = {}

    async def initialize(self) -> None:
        """Initialize pipeline from configuration."""
        sorted_steps = sorted(self.config.steps.items(), key=lambda x: x[1].order)
        for step_name, step_config in sorted_steps:
            if not step_config.enabled:
                continue
            module_name, class_name = step_config.class_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            step_class = getattr(module, class_name)
            step_instance = step_class(**step_config.parameters)
            self.steps[step_name] = step_instance

    async def execute(self) -> None:
        """Execute pipeline according to configuration."""
        for step_name, step in self.steps.items():
            step_config = self.config.steps[step_name]
            inputs = {}
            for input_name, input_ref in step_config.inputs.items():
                ref_step, ref_output = input_ref.split('.')
                if ref_step in self.results:
                    inputs[input_name] = self.results[ref_step][ref_output]
                else:
                    raise ValueError(f'Missing input: {input_ref}')
            result = await step.execute(**inputs)
            self.results[step_name] = result

def example_usage() -> None:
    """Examples of configuration-driven architecture."""
    config = ConfigLoader.load_from_file('config/pipeline.yaml')
    print(f'Loaded pipeline: {config.name} v{config.version}')
    builder = ConfigBuilder('MyPipeline', '1.0.0')
    config = builder.with_description('Example trading pipeline').add_step('data_loader', 'src.steps.DataLoader', parameters={'symbol': 'BTCUSDT', 'timeframe': '1h'}, outputs=['data']).add_step('feature_extractor', 'src.steps.FeatureExtractor', inputs={'data': 'data_loader.data'}, parameters={'indicators': ['RSI', 'MACD']}, outputs=['features']).add_dependency('logger', 'ConsoleLogger', 'src.logging', type='singleton').with_global_setting('debug', True).build()
    builder.save('config/generated_pipeline.yaml', ConfigFormat.YAML)
    pipeline = ConfigDrivenPipeline(config)
if __name__ == '__main__':
    with open('config/example_pipeline.yaml', 'w') as f:
        f.write(EXAMPLE_YAML_CONFIG)
    example_usage()