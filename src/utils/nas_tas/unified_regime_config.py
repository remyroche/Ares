"""
Unified Regime Configuration System

Provides a unified configuration system that combines the best features
of both TAS and NAS regime detection systems.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RegimeSystemType(Enum):
    """Types of regime detection systems."""
    TAS = "tas"
    NAS = "nas"
    HYBRID = "hybrid"
    UNIFIED = "unified"

class ArchitectureType(Enum):
    """Types of architectures for regime detection."""
    TREE_BASED = "tree_based"
    NEURAL_ODE = "neural_ode"
    VISION_TRANSFORMER = "vision_transformer"
    STATE_SPACE_MODEL = "state_space_model"
    HYBRID = "hybrid"
    EVOLUTIONARY = "evolutionary"

class SearchStrategy(Enum):
    """NAS search strategies."""
    EVOLUTIONARY = "evolutionary"
    BAYESIAN = "bayesian"
    RANDOM = "random"
    META_LEARNING = "meta_learning"
    TREE_BASED = "tree_based"

class OptimizationLevel(Enum):
    """Optimization levels for regime detection."""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    MAXIMUM = "maximum"

@dataclass
class EconomicEvaluationConfig:
    """Configuration for economic significance evaluation."""
    price_impact_weight: float = 0.3
    volume_significance_weight: float = 0.2
    volatility_impact_weight: float = 0.2
    trend_consistency_weight: float = 0.15
    market_efficiency_weight: float = 0.15
    significance_threshold: float = 0.7
    economic_indicators: List[str] = field(default_factory=lambda: [
        'gdp_growth', 'inflation_rate', 'interest_rate', 'unemployment_rate'
    ])

@dataclass
class TradingViabilityConfig:
    """Configuration for trading viability evaluation."""
    minimum_regime_duration: int = 15  # minutes
    maximum_regime_duration: int = 180  # minutes
    volatility_threshold: float = 0.02
    volume_threshold: float = 1.5
    trend_strength_threshold: float = 0.6
    liquidity_threshold: float = 0.8
    viability_threshold: float = 0.6

@dataclass
class HardwareOptimizationConfig:
    """Configuration for hardware optimization."""
    enable_gpu_acceleration: bool = True
    enable_mixed_precision: bool = True
    enable_memory_optimization: bool = True
    max_memory_usage_gb: float = 8.0
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning."""
    meta_learning_rate: float = 1e-3
    inner_learning_rate: float = 0.01
    num_inner_steps: int = 5
    num_outer_steps: int = 100
    num_shots: int = 5
    num_ways: int = 5
    adaptation_steps: int = 10
    use_uncertainty: bool = True
    memory_size: int = 1000

@dataclass
class UnifiedRegimeConfig:
    """Unified configuration for regime detection systems."""
    
    # System identification
    system_name: str = "Unified Regime Detection System"
    version: str = "1.0.0"
    system_type: RegimeSystemType = RegimeSystemType.UNIFIED
    
    # Core architecture settings
    primary_architecture: ArchitectureType = ArchitectureType.HYBRID
    enable_tree_based: bool = True
    enable_neural_odes: bool = True
    enable_vision_transformers: bool = True
    enable_state_space_models: bool = True
    enable_meta_learning: bool = True
    
    # Search configuration
    search_strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    
    # Regime detection settings
    n_regimes: int = 8
    min_regime_duration: int = 15
    max_regime_duration: int = 180
    enable_micro_regime_detection: bool = True
    micro_regime_sensitivity: float = 0.7
    
    # Timeframe settings
    primary_timeframe: str = "15m"
    micro_timeframe: str = "5m"
    sequence_length: int = 100
    
    # Tree-specific settings (for TAS)
    tree_depth: int = 6
    n_estimators: int = 1000
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: Union[str, float] = 'sqrt'
    
    # Statistical methods
    enable_statistical_methods: bool = True
    statistical_significance_level: float = 0.05
    enable_bootstrap_analysis: bool = True
    bootstrap_iterations: int = 1000
    
    # Advanced features
    enable_clvsa_enhancement: bool = True
    enable_regime_adaptation: bool = True
    enable_uncertainty_quantification: bool = True
    enable_multi_scale_analysis: bool = True
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    enable_matrix_optimization: bool = True
    enable_memory_optimization: bool = True
    optimization_level: OptimizationLevel = OptimizationLevel.MAXIMUM
    
    # Economic evaluation
    enable_economic_evaluation: bool = True
    economic_significance_threshold: float = 0.7
    trading_viability_threshold: float = 0.6
    risk_adjusted_return_threshold: float = 0.1
    
    # Performance thresholds
    accuracy_threshold: float = 0.9
    regime_stability_threshold: float = 0.8
    transition_accuracy_threshold: float = 0.85
    
    # Execution settings
    max_execution_time: int = 300  # seconds
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    enable_checkpointing: bool = True
    checkpoint_interval: int = 10
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_profiling: bool = True
    enable_visualization: bool = True
    save_results: bool = True
    results_directory: str = "unified_regime_results"
    
    # Component configurations
    economic_config: EconomicEvaluationConfig = field(default_factory=EconomicEvaluationConfig)
    trading_config: TradingViabilityConfig = field(default_factory=TradingViabilityConfig)
    hardware_config: HardwareOptimizationConfig = field(default_factory=HardwareOptimizationConfig)
    meta_learning_config: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    
    # Multi-timeframe settings
    trading_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    regime_detection_timeframe: str = "15m"
    enable_multi_timeframe_training: bool = True
    
    # Integration settings
    enable_tas_integration: bool = True
    enable_nas_integration: bool = True
    tas_base_weight: float = 0.4
    nas_base_weight: float = 0.4
    adaptive_weighting: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_logging()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        try:
            # Validate thresholds
            if not (0.0 <= self.accuracy_threshold <= 1.0):
                raise ValueError(f"Invalid accuracy threshold: {self.accuracy_threshold}")
            
            if not (0.0 <= self.economic_significance_threshold <= 1.0):
                raise ValueError(f"Invalid economic significance threshold: {self.economic_significance_threshold}")
            
            if not (0.0 <= self.trading_viability_threshold <= 1.0):
                raise ValueError(f"Invalid trading viability threshold: {self.trading_viability_threshold}")
            
            # Validate timeframes
            if self.min_regime_duration >= self.max_regime_duration:
                raise ValueError("Minimum regime duration must be less than maximum")
            
            # Validate tree parameters
            if self.tree_depth < 3 or self.tree_depth > 15:
                raise ValueError("tree_depth must be between 3 and 15")
            
            if self.n_estimators < 100 or self.n_estimators > 5000:
                raise ValueError("n_estimators must be between 100 and 5000")
            
            logger.info("✅ Unified configuration validation passed")
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @classmethod
    def create_tas_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration optimized for TAS regime detection."""
        config = cls()
        config.system_type = RegimeSystemType.TAS
        config.primary_architecture = ArchitectureType.TREE_BASED
        config.enable_tree_based = True
        config.enable_neural_odes = False
        config.enable_vision_transformers = False
        config.search_strategy = SearchStrategy.TREE_BASED
        config.n_regimes = 8
        config.primary_timeframe = "15m"
        config.tree_depth = 6
        config.n_estimators = 1000
        config.enable_statistical_methods = True
        config.enable_bootstrap_analysis = True
        return config
    
    @classmethod
    def create_nas_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration optimized for NAS regime detection."""
        config = cls()
        config.system_type = RegimeSystemType.NAS
        config.primary_architecture = ArchitectureType.HYBRID
        config.enable_tree_based = False
        config.enable_neural_odes = True
        config.enable_vision_transformers = True
        config.search_strategy = SearchStrategy.EVOLUTIONARY
        config.n_regimes = 10
        config.primary_timeframe = "15m"
        config.population_size = 50
        config.generations = 100
        config.enable_meta_learning = True
        return config
    
    @classmethod
    def create_hybrid_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration for hybrid TAS-NAS regime detection."""
        config = cls()
        config.system_type = RegimeSystemType.HYBRID
        config.primary_architecture = ArchitectureType.HYBRID
        config.enable_tree_based = True
        config.enable_neural_odes = True
        config.enable_vision_transformers = True
        config.search_strategy = SearchStrategy.EVOLUTIONARY
        config.n_regimes = 12
        config.primary_timeframe = "15m"
        config.enable_tas_integration = True
        config.enable_nas_integration = True
        config.adaptive_weighting = True
        return config
    
    @classmethod
    def create_short_term_trading_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration optimized for short-term trading."""
        config = cls()
        config.primary_timeframe = "15m"
        config.micro_timeframe = "5m"
        config.n_regimes = 12
        config.min_regime_duration = 15
        config.max_regime_duration = 180
        config.enable_micro_regime_detection = True
        config.micro_regime_sensitivity = 0.7
        config.economic_config.significance_threshold = 0.7
        config.trading_config.viability_threshold = 0.6
        config.max_execution_time = 120
        return config
    
    @classmethod
    def create_research_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration for research and experimentation."""
        config = cls()
        config.primary_architecture = ArchitectureType.HYBRID
        config.enable_tree_based = True
        config.enable_neural_odes = True
        config.enable_vision_transformers = True
        config.enable_meta_learning = True
        config.population_size = 100
        config.generations = 200
        config.enable_profiling = True
        config.enable_visualization = True
        config.max_execution_time = 600
        return config
    
    @classmethod
    def create_production_config(cls) -> 'UnifiedRegimeConfig':
        """Create configuration for production deployment."""
        config = cls()
        config.primary_architecture = ArchitectureType.EVOLUTIONARY
        config.population_size = 30
        config.generations = 50
        config.max_execution_time = 120
        config.enable_early_stopping = True
        config.hardware_config.enable_gpu_acceleration = True
        config.hardware_config.enable_mixed_precision = True
        config.log_level = "WARNING"
        config.enable_profiling = False
        return config
    
    def get_architecture_config(self) -> Dict[str, Any]:
        """Get architecture-specific configuration."""
        config = {
            'tree_based': {
                'enabled': self.enable_tree_based,
                'tree_depth': self.tree_depth,
                'n_estimators': self.n_estimators,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features
            } if self.enable_tree_based else None,
            'neural_ode': {
                'enabled': self.enable_neural_odes,
                'state_size': 64,
                'hidden_size': 128,
                'time_points': 20
            } if self.enable_neural_odes else None,
            'vision_transformer': {
                'enabled': self.enable_vision_transformers,
                'sequence_length': self.sequence_length,
                'embed_dim': 64,
                'num_heads': 8,
                'num_layers': 6
            } if self.enable_vision_transformers else None,
            'primary_architecture': self.primary_architecture.value,
            'hybrid_mode': self.primary_architecture == ArchitectureType.HYBRID
        }
        return config
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration."""
        return {
            'strategy': self.search_strategy.value,
            'population_size': self.population_size,
            'generations': self.generations,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'elite_size': self.elite_size
        }
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return {
            'economic': self.economic_config.__dict__,
            'trading': self.trading_config.__dict__,
            'thresholds': {
                'accuracy': self.accuracy_threshold,
                'economic_significance': self.economic_significance_threshold,
                'trading_viability': self.trading_viability_threshold,
                'regime_stability': self.regime_stability_threshold,
                'transition_accuracy': self.transition_accuracy_threshold
            }
        }
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware optimization configuration."""
        return self.hardware_config.__dict__
    
    def get_meta_learning_config(self) -> Dict[str, Any]:
        """Get meta-learning configuration."""
        return self.meta_learning_config.__dict__