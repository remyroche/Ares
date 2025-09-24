"""
Perfect NAS Regime System Configuration

Unified configuration system that combines the best of both nas_modeling and nas_clustering
systems with enhanced economic and trading focus.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class NeuralArchitectureType(Enum):
    """Types of neural architectures for regime detection."""
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

class OptimizationObjective(Enum):
    """Multi-objective optimization objectives."""
    REGIME_ACCURACY = "regime_accuracy"
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    TRADING_VIABILITY = "trading_viability"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    ARCHITECTURE_COMPLEXITY = "architecture_complexity"
    REGIME_STABILITY = "regime_stability"
    TRANSITION_ACCURACY = "transition_accuracy"

@dataclass
class NeuralODEConfig:
    """Configuration for Neural ODEs."""
    state_size: int = 64
    hidden_size: int = 128
    time_points: int = 20
    method: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-6
    use_adjoint: bool = True
    event_detection: bool = True
    adaptive_stepping: bool = True

@dataclass
class VisionTransformerConfig:
    """Configuration for Vision Transformers."""
    sequence_length: int = 100
    feature_dim: int = 4
    patch_size: int = 10
    embed_dim: int = 64
    num_heads: int = 8
    num_layers: int = 6
    dropout_rate: float = 0.1
    use_positional_encoding: bool = True

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
class PerfectNASConfig:
    """Perfect NAS Regime System Configuration."""
    
    # System identification
    system_name: str = "Perfect NAS Regime System"
    version: str = "1.0.0"
    
    # Core architecture settings
    primary_architecture: NeuralArchitectureType = NeuralArchitectureType.HYBRID
    enable_neural_odes: bool = True
    enable_vision_transformers: bool = True
    enable_state_space_models: bool = True
    enable_meta_learning: bool = True
    
    # NAS search configuration
    search_strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    
    # Multi-objective optimization
    objectives: List[OptimizationObjective] = field(default_factory=lambda: [
        OptimizationObjective.REGIME_ACCURACY,
        OptimizationObjective.ECONOMIC_SIGNIFICANCE,
        OptimizationObjective.TRADING_VIABILITY,
        OptimizationObjective.COMPUTATIONAL_EFFICIENCY,
        OptimizationObjective.ARCHITECTURE_COMPLEXITY
    ])
    objective_weights: Dict[OptimizationObjective, float] = field(default_factory=lambda: {
        OptimizationObjective.REGIME_ACCURACY: 0.3,
        OptimizationObjective.ECONOMIC_SIGNIFICANCE: 0.25,
        OptimizationObjective.TRADING_VIABILITY: 0.25,
        OptimizationObjective.COMPUTATIONAL_EFFICIENCY: 0.1,
        OptimizationObjective.ARCHITECTURE_COMPLEXITY: 0.1
    })
    
    # Regime detection settings
    n_regimes: int = 10
    min_regime_duration: int = 15
    max_regime_duration: int = 180
    enable_micro_regime_detection: bool = True
    micro_regime_sensitivity: float = 0.7
    
    # Timeframe settings
    primary_timeframe: str = "15m"
    micro_timeframe: str = "5m"
    sequence_length: int = 100
    
    # Component configurations
    neural_ode_config: NeuralODEConfig = field(default_factory=NeuralODEConfig)
    vision_transformer_config: VisionTransformerConfig = field(default_factory=VisionTransformerConfig)
    meta_learning_config: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    economic_config: EconomicEvaluationConfig = field(default_factory=EconomicEvaluationConfig)
    trading_config: TradingViabilityConfig = field(default_factory=TradingViabilityConfig)
    hardware_config: HardwareOptimizationConfig = field(default_factory=HardwareOptimizationConfig)
    
    # Performance thresholds
    accuracy_threshold: float = 0.9
    economic_significance_threshold: float = 0.8
    trading_viability_threshold: float = 0.7
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
    results_directory: str = "perfect_nas_results"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_logging()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        try:
            # Validate objective weights sum to 1.0
            total_weight = sum(self.objective_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                logger.warning(f"Objective weights sum to {total_weight}, normalizing to 1.0")
                for obj in self.objective_weights:
                    self.objective_weights[obj] /= total_weight
            
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
            
            logger.info("✅ Configuration validation passed")
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_architecture_config(self) -> Dict[str, Any]:
        """Get architecture-specific configuration."""
        config = {
            'neural_ode': self.neural_ode_config.__dict__ if self.enable_neural_odes else None,
            'vision_transformer': self.vision_transformer_config.__dict__ if self.enable_vision_transformers else None,
            'meta_learning': self.meta_learning_config.__dict__ if self.enable_meta_learning else None,
            'primary_architecture': self.primary_architecture.value,
            'hybrid_mode': self.primary_architecture == NeuralArchitectureType.HYBRID
        }
        return config
    
    def get_nas_search_config(self) -> Dict[str, Any]:
        """Get NAS search configuration."""
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
            'objectives': [obj.value for obj in self.objectives],
            'weights': {obj.value: weight for obj, weight in self.objective_weights.items()},
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
    
    def create_short_term_trading_config() -> 'PerfectNASConfig':
        """Create configuration optimized for short-term trading."""
        config = PerfectNASConfig()
        config.primary_timeframe = "15m"
        config.micro_timeframe = "5m"
        config.n_regimes = 12
        config.min_regime_duration = 15
        config.max_regime_duration = 180
        config.enable_micro_regime_detection = True
        config.micro_regime_sensitivity = 0.7
        config.economic_config.significance_threshold = 0.7
        config.trading_config.viability_threshold = 0.6
        return config
    
    def create_research_config() -> 'PerfectNASConfig':
        """Create configuration optimized for research and experimentation."""
        config = PerfectNASConfig()
        config.primary_architecture = NeuralArchitectureType.HYBRID
        config.enable_neural_odes = True
        config.enable_vision_transformers = True
        config.enable_meta_learning = True
        config.population_size = 100
        config.generations = 200
        config.enable_profiling = True
        config.enable_visualization = True
        return config
    
    def create_production_config() -> 'PerfectNASConfig':
        """Create configuration optimized for production deployment."""
        config = PerfectNASConfig()
        config.primary_architecture = NeuralArchitectureType.EVOLUTIONARY
        config.population_size = 30
        config.generations = 50
        config.max_execution_time = 120
        config.enable_early_stopping = True
        config.hardware_config.enable_gpu_acceleration = True
        config.hardware_config.enable_mixed_precision = True
        config.log_level = "WARNING"
        config.enable_profiling = False
        return config