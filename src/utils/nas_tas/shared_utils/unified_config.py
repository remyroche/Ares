"""
Unified Configuration System for NAS and TAS Systems

This module provides a unified configuration system that consolidates
all configuration parameters for both Neural Architecture Search (NAS)
and Tree Architecture Search (TAS) systems.
"""

import json
import yaml
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, TypeVar
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ConfigFormat(Enum):
    """Supported configuration formats."""
    JSON = "json"
    YAML = "yaml"
    PICKLE = "pickle"
    PYTHON = "python"

class ArchitectureType(Enum):
    """Types of architectures supported."""
    NEURAL = "neural"
    TREE = "tree"
    HYBRID = "hybrid"

@dataclass
class BaseConfig(ABC):
    """Base configuration class."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """Convert configuration to JSON."""
        json_str = json.dumps(self.to_dict(), indent=2, default=str)
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def to_yaml(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """Convert configuration to YAML."""
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False)
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(yaml_str)
        
        return yaml_str
    
    def to_pickle(self, file_path: Union[str, Path]):
        """Save configuration as pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls: Type[T], file_path: Union[str, Path]) -> T:
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls: Type[T], file_path: Union[str, Path]) -> T:
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_pickle(cls: Type[T], file_path: Union[str, Path]) -> T:
        """Load configuration from pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)

@dataclass
class SearchConfig(BaseConfig):
    """Unified search configuration."""
    
    # Core search parameters
    architecture_type: ArchitectureType = ArchitectureType.HYBRID
    max_iterations: int = 100
    population_size: int = 50
    elite_size: int = 5
    
    # Search strategy parameters
    search_strategy: str = "enhanced_bayesian"
    bayesian_config: Dict[str, Any] = field(default_factory=lambda: {
        'n_initial_points': 10,
        'acquisition_function': 'ei',
        'random_state': 42
    })
    
    evolutionary_config: Dict[str, Any] = field(default_factory=lambda: {
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'tournament_size': 3
    })
    
    # Hardware optimization
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    enable_gpu_acceleration: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Monitoring and logging
    enable_logging: bool = True
    log_level: str = 'INFO'
    save_intermediate_results: bool = True
    checkpoint_frequency: int = 10

@dataclass
class OptimizationConfig(BaseConfig):
    """Unified optimization configuration."""
    
    # Core optimization parameters
    algorithm: str = "nsga2"
    objectives: List[str] = field(default_factory=lambda: [
        "accuracy", "efficiency", "profitability"
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    
    # Optimization parameters
    max_iterations: int = 100
    population_size: int = 50
    convergence_threshold: float = 0.01
    convergence_patience: int = 20
    
    # Algorithm-specific parameters
    nsga2_config: Dict[str, Any] = field(default_factory=lambda: {
        'crossover_probability': 0.8,
        'mutation_probability': 0.1,
        'tournament_size': 3,
        'eta_c': 20,
        'eta_m': 20
    })
    
    bayesian_config: Dict[str, Any] = field(default_factory=lambda: {
        'n_initial_points': 10,
        'acquisition_function': 'ei',
        'random_state': 42
    })
    
    # Hardware optimization
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    enable_gpu_acceleration: bool = True

@dataclass
class EvaluationConfig(BaseConfig):
    """Unified evaluation configuration."""
    
    # Core evaluation parameters
    evaluation_types: List[str] = field(default_factory=lambda: [
        "economic_significance",
        "trading_viability",
        "performance_metrics"
    ])
    architecture_type: ArchitectureType = ArchitectureType.HYBRID
    
    # Economic significance parameters
    significance_threshold: float = 0.05
    min_regime_duration: int = 10
    volatility_threshold: float = 0.3
    efficiency_threshold: float = 0.6
    
    # Trading viability parameters
    min_trading_frequency: float = 0.1
    max_trading_frequency: float = 10.0
    min_win_rate: float = 0.4
    min_profit_factor: float = 1.1
    
    # Performance metrics parameters
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95
    lookback_period: int = 252
    
    # Monitoring and logging
    enable_logging: bool = True
    log_level: str = 'INFO'
    save_evaluation_results: bool = True

@dataclass
class RegimeDetectionConfig(BaseConfig):
    """Unified regime detection configuration."""
    
    # Core detection parameters
    method: str = "hybrid"
    architecture_type: ArchitectureType = ArchitectureType.HYBRID
    n_regimes: int = 3
    min_regime_duration: int = 10
    
    # Clustering parameters
    clustering_algorithm: str = "kmeans"
    n_clusters: int = 3
    random_state: int = 42
    
    # HMM parameters
    hmm_n_states: int = 3
    hmm_n_iterations: int = 100
    
    # Change point detection parameters
    change_point_method: str = "pelt"
    change_point_penalty: float = 1.0
    
    # Neural network parameters
    neural_n_epochs: int = 100
    neural_hidden_size: int = 64
    
    # Tree parameters
    tree_max_depth: int = 10
    tree_min_samples_split: int = 2
    
    # Advanced parameters
    enable_regime_validation: bool = True
    stability_threshold: float = 0.7
    separation_threshold: float = 0.5

@dataclass
class UtilityConfig(BaseConfig):
    """Unified utility configuration."""
    
    # Data processing parameters
    enable_data_validation: bool = True
    enable_memory_optimization: bool = True
    enable_hardware_optimization: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Validation parameters
    strict_validation: bool = False
    auto_fix_issues: bool = True
    validation_threshold: float = 0.95
    
    # Performance parameters
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    
    # Logging parameters
    enable_logging: bool = True
    log_level: str = 'INFO'
    enable_progress_tracking: bool = True

@dataclass
class UnifiedConfig(BaseConfig):
    """Unified configuration for the entire system."""
    
    # Component configurations
    search_config: SearchConfig = field(default_factory=SearchConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    regime_detection_config: RegimeDetectionConfig = field(default_factory=RegimeDetectionConfig)
    utility_config: UtilityConfig = field(default_factory=UtilityConfig)
    
    # System-wide parameters
    system_name: str = "Unified NAS-TAS System"
    version: str = "1.0.0"
    environment: str = "development"
    
    # Global settings
    enable_logging: bool = True
    log_level: str = 'INFO'
    log_file: Optional[str] = None
    
    # Data settings
    data_directory: str = "./data"
    results_directory: str = "./results"
    cache_directory: str = "./cache"
    
    # Performance settings
    max_memory_usage_gb: float = 8.0
    enable_profiling: bool = False
    profiling_output_file: Optional[str] = None
    
    def get_component_config(self, component_name: str) -> Optional[BaseConfig]:
        """Get configuration for a specific component."""
        config_map = {
            'search': self.search_config,
            'optimization': self.optimization_config,
            'evaluation': self.evaluation_config,
            'regime_detection': self.regime_detection_config,
            'utility': self.utility_config
        }
        
        return config_map.get(component_name)
    
    def update_component_config(self, component_name: str, config: BaseConfig):
        """Update configuration for a specific component."""
        if component_name == 'search':
            self.search_config = config
        elif component_name == 'optimization':
            self.optimization_config = config
        elif component_name == 'evaluation':
            self.evaluation_config = config
        elif component_name == 'regime_detection':
            self.regime_detection_config = config
        elif component_name == 'utility':
            self.utility_config = config
        else:
            raise ValueError(f"Unknown component: {component_name}")
    
    def validate(self) -> Dict[str, Any]:
        """Validate the entire configuration."""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Validate individual components
        components = [
            ('search', self.search_config),
            ('optimization', self.optimization_config),
            ('evaluation', self.evaluation_config),
            ('regime_detection', self.regime_detection_config),
            ('utility', self.utility_config)
        ]
        
        for component_name, config in components:
            component_validation = self._validate_component(component_name, config)
            
            if not component_validation['is_valid']:
                validation_result['is_valid'] = False
                validation_result['errors'].extend(component_validation['errors'])
            
            validation_result['warnings'].extend(component_validation['warnings'])
        
        # System-wide validation
        system_validation = self._validate_system_wide()
        if not system_validation['is_valid']:
            validation_result['is_valid'] = False
            validation_result['errors'].extend(system_validation['errors'])
        
        validation_result['warnings'].extend(system_validation['warnings'])
        
        return validation_result
    
    def _validate_component(self, component_name: str, config: BaseConfig) -> Dict[str, Any]:
        """Validate a specific component configuration."""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Component-specific validation
        if component_name == 'search':
            if config.max_iterations <= 0:
                validation_result['errors'].append("max_iterations must be positive")
            
            if config.population_size <= 0:
                validation_result['errors'].append("population_size must be positive")
            
            if config.population_size > 10000:
                validation_result['warnings'].append("Large population size may impact performance")
        
        elif component_name == 'optimization':
            if len(config.objectives) != len(config.objective_weights):
                validation_result['errors'].append("objectives and objective_weights must have same length")
            
            if abs(sum(config.objective_weights) - 1.0) > 0.01:
                validation_result['warnings'].append("objective_weights should sum to 1.0")
        
        elif component_name == 'evaluation':
            if config.significance_threshold <= 0 or config.significance_threshold >= 1:
                validation_result['errors'].append("significance_threshold must be between 0 and 1")
        
        elif component_name == 'regime_detection':
            if config.n_regimes <= 0:
                validation_result['errors'].append("n_regimes must be positive")
            
            if config.min_regime_duration <= 0:
                validation_result['errors'].append("min_regime_duration must be positive")
        
        elif component_name == 'utility':
            if config.validation_threshold <= 0 or config.validation_threshold >= 1:
                validation_result['errors'].append("validation_threshold must be between 0 and 1")
        
        validation_result['is_valid'] = len(validation_result['errors']) == 0
        
        return validation_result
    
    def _validate_system_wide(self) -> Dict[str, Any]:
        """Validate system-wide configuration."""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check memory usage
        total_memory = 0
        if self.search_config.memory_limit_gb:
            total_memory += self.search_config.memory_limit_gb
        if self.utility_config.memory_limit_gb:
            total_memory += self.utility_config.memory_limit_gb
        
        if total_memory > self.max_memory_usage_gb:
            validation_result['warnings'].append(
                f"Total memory usage ({total_memory}GB) exceeds system limit ({self.max_memory_usage_gb}GB)"
            )
        
        # Check directory paths
        directories = [self.data_directory, self.results_directory, self.cache_directory]
        for directory in directories:
            if not Path(directory).parent.exists():
                validation_result['warnings'].append(f"Directory {directory} does not exist")
        
        return validation_result
    
    def merge_config(self, other_config: 'UnifiedConfig', 
                    overwrite: bool = True) -> 'UnifiedConfig':
        """Merge with another configuration."""
        merged_config = UnifiedConfig()
        
        # Merge component configurations
        components = ['search', 'optimization', 'evaluation', 'regime_detection', 'utility']
        
        for component in components:
            current_config = getattr(self, f"{component}_config")
            other_component_config = getattr(other_config, f"{component}_config")
            
            if overwrite:
                # Overwrite with other config
                setattr(merged_config, f"{component}_config", other_component_config)
            else:
                # Merge configurations
                merged_component = self._merge_component_configs(current_config, other_component_config)
                setattr(merged_config, f"{component}_config", merged_component)
        
        # Merge system-wide parameters
        merged_config.system_name = other_config.system_name if overwrite else self.system_name
        merged_config.version = other_config.version if overwrite else self.version
        merged_config.environment = other_config.environment if overwrite else self.environment
        
        return merged_config
    
    def _merge_component_configs(self, config1: BaseConfig, config2: BaseConfig) -> BaseConfig:
        """Merge two component configurations."""
        # Simple merge - use config2 values where they exist, otherwise use config1
        config1_dict = config1.to_dict()
        config2_dict = config2.to_dict()
        
        merged_dict = {**config1_dict, **config2_dict}
        
        # Create new instance of the same type as config1
        return config1.__class__.from_dict(merged_dict)
    
    @classmethod
    def create_default(cls) -> 'UnifiedConfig':
        """Create default unified configuration."""
        return cls()
    
    @classmethod
    def create_from_file(cls, file_path: Union[str, Path], 
                        format: ConfigFormat = ConfigFormat.JSON) -> 'UnifiedConfig':
        """Create configuration from file."""
        if format == ConfigFormat.JSON:
            return cls.from_json(file_path)
        elif format == ConfigFormat.YAML:
            return cls.from_yaml(file_path)
        elif format == ConfigFormat.PICKLE:
            return cls.from_pickle(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def save_to_file(self, file_path: Union[str, Path], 
                    format: ConfigFormat = ConfigFormat.JSON):
        """Save configuration to file."""
        if format == ConfigFormat.JSON:
            self.to_json(file_path)
        elif format == ConfigFormat.YAML:
            self.to_yaml(file_path)
        elif format == ConfigFormat.PICKLE:
            self.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

class ConfigManager:
    """Configuration manager for handling multiple configurations."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self.configurations: Dict[str, UnifiedConfig] = {}
        self.current_config: Optional[str] = None
        
        # Load default configuration
        self.configurations['default'] = UnifiedConfig.create_default()
        self.current_config = 'default'
    
    def add_config(self, name: str, config: UnifiedConfig):
        """Add a configuration."""
        self.configurations[name] = config
    
    def get_config(self, name: Optional[str] = None) -> UnifiedConfig:
        """Get a configuration by name."""
        config_name = name or self.current_config
        
        if config_name not in self.configurations:
            raise ValueError(f"Configuration '{config_name}' not found")
        
        return self.configurations[config_name]
    
    def set_current_config(self, name: str):
        """Set the current configuration."""
        if name not in self.configurations:
            raise ValueError(f"Configuration '{name}' not found")
        
        self.current_config = name
    
    def list_configs(self) -> List[str]:
        """List all available configurations."""
        return list(self.configurations.keys())
    
    def remove_config(self, name: str):
        """Remove a configuration."""
        if name in self.configurations:
            del self.configurations[name]
            
            if self.current_config == name:
                self.current_config = None
    
    def create_config_from_template(self, template_name: str, new_name: str) -> UnifiedConfig:
        """Create a new configuration from a template."""
        if template_name not in self.configurations:
            raise ValueError(f"Template configuration '{template_name}' not found")
        
        template_config = self.configurations[template_name]
        new_config = UnifiedConfig()
        
        # Copy template configuration
        new_config.search_config = template_config.search_config
        new_config.optimization_config = template_config.optimization_config
        new_config.evaluation_config = template_config.evaluation_config
        new_config.regime_detection_config = template_config.regime_detection_config
        new_config.utility_config = template_config.utility_config
        
        # Add new configuration
        self.configurations[new_name] = new_config
        
        return new_config

# Global configuration manager instance
config_manager = ConfigManager()

# Convenience functions
def get_config(name: Optional[str] = None) -> UnifiedConfig:
    """Get configuration from global manager."""
    return config_manager.get_config(name)

def set_config(name: str, config: UnifiedConfig):
    """Set configuration in global manager."""
    config_manager.add_config(name, config)
    config_manager.set_current_config(name)

def create_default_config() -> UnifiedConfig:
    """Create default configuration."""
    return UnifiedConfig.create_default()

def load_config_from_file(file_path: Union[str, Path], 
                         format: ConfigFormat = ConfigFormat.JSON) -> UnifiedConfig:
    """Load configuration from file."""
    return UnifiedConfig.create_from_file(file_path, format)

# Export main classes and functions
__all__ = [
    'UnifiedConfig',
    'SearchConfig',
    'OptimizationConfig',
    'EvaluationConfig',
    'RegimeDetectionConfig',
    'UtilityConfig',
    'ConfigManager',
    'ConfigFormat',
    'ArchitectureType',
    'config_manager',
    'get_config',
    'set_config',
    'create_default_config',
    'load_config_from_file'
]