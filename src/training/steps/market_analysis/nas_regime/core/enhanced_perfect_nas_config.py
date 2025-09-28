"""
Enhanced Perfect NAS Configuration with Adaptive Thresholds

Extends the original configuration to include data-driven threshold learning
instead of hardcoded values.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import logging
import numpy as np
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)

from .perfect_nas_config import (
    PerfectNASConfig, NeuralArchitectureType, SearchStrategy, OptimizationObjective,
    NeuralODEConfig, VisionTransformerConfig, MetaLearningConfig,
    EconomicEvaluationConfig, TradingViabilityConfig, HardwareOptimizationConfig
)
from .adaptive_threshold_learning import (
    AdaptiveThresholdLearner, ThresholdLearningConfig, AdaptiveThresholds
)

logger = logging.getLogger(__name__)

class ThresholdLearningMode(Enum):
    """Modes for threshold learning."""
    DISABLED = "disabled"  # Use hardcoded thresholds
    LEARNING = "learning"  # Learn from historical data
    ADAPTIVE = "adaptive"  # Continuously adapt thresholds
    HYBRID = "hybrid"  # Combine learned and hardcoded thresholds

@dataclass
class AdaptiveThresholdConfig:
    """Configuration for adaptive threshold learning."""
    # Learning mode
    learning_mode: ThresholdLearningMode = ThresholdLearningMode.ADAPTIVE
    
    # Learning parameters
    enable_economic_learning: bool = True
    enable_trading_learning: bool = True
    enable_stability_learning: bool = True
    
    # Learning frequency
    learning_frequency: int = 100  # Learn every N samples
    min_samples_for_learning: int = 200  # Minimum samples required
    lookback_periods: int = 1000  # Historical periods for learning
    
    # Threshold bounds (safety limits)
    economic_bounds: Tuple[float, float] = (0.3, 0.95)
    trading_bounds: Tuple[float, float] = (0.2, 0.9)
    stability_bounds: Tuple[float, float] = (0.4, 0.95)
    
    # Market condition adaptation
    enable_volatility_adaptation: bool = True
    enable_liquidity_adaptation: bool = True
    enable_stress_adaptation: bool = True
    enable_trend_adaptation: bool = True
    
    # Fallback behavior
    use_hardcoded_fallback: bool = True
    hardcoded_economic: float = 0.8
    hardcoded_trading: float = 0.7
    hardcoded_stability: float = 0.8
    
    # Learning confidence
    min_learning_confidence: float = 0.6
    confidence_decay_rate: float = 0.95

@dataclass
class EnhancedPerfectNASConfig(PerfectNASConfig):
    """
    Enhanced Perfect NAS Configuration with adaptive threshold learning.
    
    Extends the original configuration to include data-driven threshold learning
    instead of hardcoded values for economic significance, trading viability,
    and regime stability.
    """
    
    # Adaptive threshold configuration
    adaptive_thresholds: AdaptiveThresholdConfig = field(default_factory=AdaptiveThresholdConfig)
    
    # Threshold learning state
    _threshold_learner: Optional[AdaptiveThresholdLearner] = None
    _learned_thresholds: Optional[AdaptiveThresholds] = None
    _threshold_learning_enabled: bool = True
    
    def __post_init__(self):
        """Initialize enhanced configuration."""
        tprint("🚀 [ENHANCED_NAS_CONFIG] Initializing Enhanced Perfect NAS Configuration", color="cyan", bold=True)
        super().__post_init__()
        tprint("🔧 [ENHANCED_NAS_CONFIG] Initializing adaptive thresholds", color="yellow")
        self._initialize_adaptive_thresholds()
        tprint("✅ [ENHANCED_NAS_CONFIG] Enhanced Perfect NAS Configuration initialized successfully", color="green")
    
    def _initialize_adaptive_thresholds(self):
        """Initialize adaptive threshold learning."""
        try:
            if self.adaptive_thresholds.learning_mode != ThresholdLearningMode.DISABLED:
                tprint(f"📊 [ENHANCED_NAS_CONFIG] Learning mode: {self.adaptive_thresholds.learning_mode.value}", color="blue")
                # Create threshold learning configuration
                learning_config = ThresholdLearningConfig(
                    lookback_periods=self.adaptive_thresholds.lookback_periods,
                    min_samples_for_learning=self.adaptive_thresholds.min_samples_for_learning,
                    learning_frequency=self.adaptive_thresholds.learning_frequency,
                    economic_significance_bounds=self.adaptive_thresholds.economic_bounds,
                    trading_viability_bounds=self.adaptive_thresholds.trading_bounds,
                    regime_stability_bounds=self.adaptive_thresholds.stability_bounds
                )
                tprint(f"📊 [ENHANCED_NAS_CONFIG] Lookback periods: {self.adaptive_thresholds.lookback_periods}", color="blue")
                tprint(f"📊 [ENHANCED_NAS_CONFIG] Learning frequency: {self.adaptive_thresholds.learning_frequency}", color="blue")
                
                # Initialize threshold learner
                tprint("🧠 [ENHANCED_NAS_CONFIG] Creating threshold learner", color="yellow")
                self._threshold_learner = AdaptiveThresholdLearner(learning_config)
                self._threshold_learning_enabled = True
                
                tprint("✅ [ENHANCED_NAS_CONFIG] Adaptive threshold learning initialized successfully", color="green")
                logger.info("✅ Adaptive threshold learning initialized")
            else:
                self._threshold_learning_enabled = False
                tprint("⚠️ [ENHANCED_NAS_CONFIG] Adaptive threshold learning disabled", color="yellow")
                logger.info("⚠️ Adaptive threshold learning disabled")
                
        except Exception as e:
            tprint(f"❌ [ENHANCED_NAS_CONFIG] Adaptive threshold initialization failed: {e}", color="red")
            logger.error(f"❌ Adaptive threshold initialization failed: {e}")
            self._threshold_learning_enabled = False
    
    def learn_thresholds(self, market_data: np.ndarray, regime_predictions: np.ndarray,
                        timestamps: Optional[np.ndarray] = None) -> bool:
        """
        Learn adaptive thresholds from historical data.
        
        Args:
            market_data: Historical market data (OHLCV)
            regime_predictions: Historical regime predictions
            timestamps: Optional timestamps
            
        Returns:
            True if learning was successful, False otherwise
        """
        try:
            if not self._threshold_learning_enabled or not self._threshold_learner:
                logger.warning("Threshold learning not enabled")
                return False
            
            if len(market_data) < self.adaptive_thresholds.min_samples_for_learning:
                logger.warning(f"Insufficient data for learning: {len(market_data)} < {self.adaptive_thresholds.min_samples_for_learning}")
                return False
            
            # Learn thresholds
            learned_thresholds = self._threshold_learner.learn_thresholds(
                market_data, regime_predictions, timestamps
            )
            
            if learned_thresholds.learning_confidence >= self.adaptive_thresholds.min_learning_confidence:
                self._learned_thresholds = learned_thresholds
                self._update_configuration_thresholds(learned_thresholds)
                logger.info("✅ Adaptive thresholds learned and applied")
                return True
            else:
                logger.warning(f"Learning confidence too low: {learned_thresholds.learning_confidence:.3f} < {self.adaptive_thresholds.min_learning_confidence}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Threshold learning failed: {e}")
            return False
    
    def _update_configuration_thresholds(self, learned_thresholds: AdaptiveThresholds):
        """Update configuration with learned thresholds."""
        try:
            # Update economic significance threshold
            if self.adaptive_thresholds.enable_economic_learning:
                self.economic_significance_threshold = learned_thresholds.economic_significance_threshold
                logger.info(f"Updated economic significance threshold: {self.economic_significance_threshold:.3f}")
            
            # Update trading viability threshold
            if self.adaptive_thresholds.enable_trading_learning:
                self.trading_viability_threshold = learned_thresholds.trading_viability_threshold
                logger.info(f"Updated trading viability threshold: {self.trading_viability_threshold:.3f}")
            
            # Update regime stability threshold
            if self.adaptive_thresholds.enable_stability_learning:
                self.regime_stability_threshold = learned_thresholds.regime_stability_threshold
                logger.info(f"Updated regime stability threshold: {self.regime_stability_threshold:.3f}")
            
            # Update economic and trading configs
            self.economic_config.significance_threshold = learned_thresholds.economic_significance_threshold
            self.trading_config.viability_threshold = learned_thresholds.trading_viability_threshold
            
        except Exception as e:
            logger.error(f"❌ Configuration update failed: {e}")
    
    def get_adaptive_thresholds(self) -> Optional[AdaptiveThresholds]:
        """Get current adaptive thresholds."""
        return self._learned_thresholds
    
    def get_threshold_explanations(self) -> Dict[str, str]:
        """Get explanations for current thresholds."""
        if self._learned_thresholds and self._threshold_learner:
            return self._threshold_learner.get_threshold_explanations()
        else:
            return {
                "economic_significance": f"Hardcoded threshold: {self.economic_significance_threshold:.3f}",
                "trading_viability": f"Hardcoded threshold: {self.trading_viability_threshold:.3f}",
                "regime_stability": f"Hardcoded threshold: {self.regime_stability_threshold:.3f}"
            }
    
    def update_thresholds(self, new_market_data: np.ndarray,
                         new_regime_predictions: np.ndarray,
                         new_timestamps: Optional[np.ndarray] = None) -> bool:
        """Update thresholds with new data."""
        try:
            if not self._threshold_learning_enabled or not self._threshold_learner:
                return False
            
            # Update thresholds
            updated_thresholds = self._threshold_learner.update_thresholds(
                new_market_data, new_regime_predictions, new_timestamps
            )
            
            if updated_thresholds.learning_confidence >= self.adaptive_thresholds.min_learning_confidence:
                self._learned_thresholds = updated_thresholds
                self._update_configuration_thresholds(updated_thresholds)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"❌ Threshold update failed: {e}")
            return False
    
    def get_effective_thresholds(self) -> Dict[str, float]:
        """Get effective thresholds (learned or fallback)."""
        if self._learned_thresholds and self.adaptive_thresholds.learning_mode != ThresholdLearningMode.DISABLED:
            return {
                'economic_significance': self._learned_thresholds.economic_significance_threshold,
                'trading_viability': self._learned_thresholds.trading_viability_threshold,
                'regime_stability': self._learned_thresholds.regime_stability_threshold
            }
        else:
            return {
                'economic_significance': self.economic_significance_threshold,
                'trading_viability': self.trading_viability_threshold,
                'regime_stability': self.regime_stability_threshold
            }
    
    def get_threshold_confidence_intervals(self) -> Dict[str, Tuple[float, float]]:
        """Get confidence intervals for thresholds."""
        if self._learned_thresholds:
            return {
                'economic_significance': self._learned_thresholds.economic_confidence_interval,
                'trading_viability': self._learned_thresholds.trading_confidence_interval,
                'regime_stability': self._learned_thresholds.stability_confidence_interval
            }
        else:
            # Return default intervals
            return {
                'economic_significance': (0.7, 0.9),
                'trading_viability': (0.6, 0.8),
                'regime_stability': (0.7, 0.9)
            }
    
    def should_learn_thresholds(self, sample_count: int) -> bool:
        """Check if thresholds should be learned based on sample count."""
        if not self._threshold_learning_enabled:
            return False
        
        return (sample_count % self.adaptive_thresholds.learning_frequency == 0 and
                sample_count >= self.adaptive_thresholds.min_samples_for_learning)
    
    def create_adaptive_short_term_trading_config() -> 'EnhancedPerfectNASConfig':
        """Create adaptive configuration for short-term trading."""
        config = EnhancedPerfectNASConfig()
        
        # Set up for short-term trading
        config.primary_timeframe = "15m"
        config.micro_timeframe = "5m"
        config.n_regimes = 12
        config.min_regime_duration = 15
        config.max_regime_duration = 180
        
        # Enable adaptive thresholds
        config.adaptive_thresholds.learning_mode = ThresholdLearningMode.ADAPTIVE
        config.adaptive_thresholds.learning_frequency = 50  # Learn more frequently
        config.adaptive_thresholds.min_samples_for_learning = 100  # Lower threshold
        
        # Enable all learning components
        config.adaptive_thresholds.enable_economic_learning = True
        config.adaptive_thresholds.enable_trading_learning = True
        config.adaptive_thresholds.enable_stability_learning = True
        
        # Enable market condition adaptation
        config.adaptive_thresholds.enable_volatility_adaptation = True
        config.adaptive_thresholds.enable_liquidity_adaptation = True
        config.adaptive_thresholds.enable_stress_adaptation = True
        config.adaptive_thresholds.enable_trend_adaptation = True
        
        return config
    
    def create_adaptive_research_config() -> 'EnhancedPerfectNASConfig':
        """Create adaptive configuration for research."""
        config = EnhancedPerfectNASConfig()
        
        # Set up for research
        config.primary_architecture = NeuralArchitectureType.HYBRID
        config.enable_neural_odes = True
        config.enable_vision_transformers = True
        config.enable_meta_learning = True
        config.population_size = 100
        config.generations = 200
        
        # Enable comprehensive adaptive thresholds
        config.adaptive_thresholds.learning_mode = ThresholdLearningMode.ADAPTIVE
        config.adaptive_thresholds.learning_frequency = 25  # Learn very frequently
        config.adaptive_thresholds.min_samples_for_learning = 50  # Very low threshold
        
        # Enable all learning and adaptation
        config.adaptive_thresholds.enable_economic_learning = True
        config.adaptive_thresholds.enable_trading_learning = True
        config.adaptive_thresholds.enable_stability_learning = True
        config.adaptive_thresholds.enable_volatility_adaptation = True
        config.adaptive_thresholds.enable_liquidity_adaptation = True
        config.adaptive_thresholds.enable_stress_adaptation = True
        config.adaptive_thresholds.enable_trend_adaptation = True
        
        return config
    
    def create_adaptive_production_config() -> 'EnhancedPerfectNASConfig':
        """Create adaptive configuration for production."""
        config = EnhancedPerfectNASConfig()
        
        # Set up for production
        config.primary_architecture = NeuralArchitectureType.EVOLUTIONARY
        config.population_size = 30
        config.generations = 50
        config.max_execution_time = 120
        
        # Enable conservative adaptive thresholds
        config.adaptive_thresholds.learning_mode = ThresholdLearningMode.HYBRID
        config.adaptive_thresholds.learning_frequency = 100  # Learn less frequently
        config.adaptive_thresholds.min_samples_for_learning = 200  # Higher threshold
        
        # Enable learning with fallback
        config.adaptive_thresholds.enable_economic_learning = True
        config.adaptive_thresholds.enable_trading_learning = True
        config.adaptive_thresholds.enable_stability_learning = True
        config.adaptive_thresholds.use_hardcoded_fallback = True
        
        # Conservative market condition adaptation
        config.adaptive_thresholds.enable_volatility_adaptation = True
        config.adaptive_thresholds.enable_liquidity_adaptation = True
        config.adaptive_thresholds.enable_stress_adaptation = False  # Disable for stability
        config.adaptive_thresholds.enable_trend_adaptation = True
        
        return config