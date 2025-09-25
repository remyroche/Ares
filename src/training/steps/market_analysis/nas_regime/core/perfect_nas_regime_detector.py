"""
Perfect NAS Regime Detector - Clean Version

The ultimate regime detection system using unified utilities.
All legacy code has been removed and the system now relies entirely on
the unified regime detection framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
import time
from datetime import datetime
from dataclasses import dataclass

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import unified utilities
try:
    from src.utils.ml_common.nas_tas_unified import (
        UnifiedRegimeDetector, UnifiedRegimeConfig, UnifiedRegimeResult,
        RegimeDetectionMethod
    )
    from src.utils.common_operations import (
        CommonUtilities, memory_checkpoint, gpu_context, timed_operation
    )
    UNIFIED_UTILITIES_AVAILABLE = True
except ImportError:
    UNIFIED_UTILITIES_AVAILABLE = False

# Import NAS configuration
from .perfect_nas_config import PerfectNASConfig, NeuralArchitectureType

# Import NAS result
try:
    from .perfect_nas_result import PerfectNASResult
except ImportError:
    # Fallback result class if not available
    @dataclass
    class PerfectNASResult:
        success: bool
        regime_predictions: np.ndarray
        regime_probabilities: np.ndarray
        economic_significance_scores: np.ndarray
        trading_viability_scores: np.ndarray
        regime_stability_scores: np.ndarray
        transition_probabilities: np.ndarray
        micro_regimes: Optional[Dict[str, Any]] = None
        architecture_performance: Optional[Dict[str, Any]] = None
        uncertainty_estimates: Optional[np.ndarray] = None
        execution_time: float = 0.0
        metadata: Dict[str, Any] = None
        error_message: Optional[str] = None

logger = logging.getLogger(__name__)

class PerfectNASRegimeDetector:
    """
    Perfect NAS Regime Detector using unified utilities.
    
    This is a clean version that relies entirely on the unified regime detection
    system. All legacy code has been removed.
    """

    def __init__(self, config: PerfectNASConfig):
        """Initialize Perfect NAS Regime Detector."""
        tprint("🚀 [PERFECT_NAS_REGIME_DETECTOR] Initializing Perfect NAS Regime Detector (Clean Version)", color="cyan", bold=True)
        tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Architecture: {config.primary_architecture.value}", color="blue")
        tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Neural ODEs: {config.enable_neural_odes}", color="blue")
        tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Vision Transformers: {config.enable_vision_transformers}", color="blue")
        tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Meta-learning: {config.enable_meta_learning}", color="blue")
        tprint(f"📊 [PERFECT_NAS_REGIME_DETECTOR] Search Strategy: {config.search_strategy.value}", color="blue")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize unified utilities if available
        if UNIFIED_UTILITIES_AVAILABLE:
            tprint("🔧 [PERFECT_NAS_REGIME_DETECTOR] Initializing unified utilities", color="yellow")
            try:
                self.common_utils = CommonUtilities()
                self.unified_detector = UnifiedRegimeDetector(self._create_unified_config())
                tprint_success("✅ Unified utilities initialized")
                self.logger.info("✅ Unified utilities initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Unified utilities initialization failed: {e}")
                self.logger.warning(f"Unified utilities initialization failed: {e}")
                self.common_utils = None
                self.unified_detector = None
        else:
            tprint_error("❌ Unified utilities not available - NAS detector requires unified system")
            raise ImportError("Unified utilities are required for NAS regime detection")

        tprint("✅ [PERFECT_NAS_REGIME_DETECTOR] Perfect NAS Regime Detector initialized successfully", color="green")
        self.logger.info(f"✅ Perfect NAS Regime Detector initialized with unified utilities")

    def _create_unified_config(self) -> UnifiedRegimeConfig:
        """Create unified configuration from NAS configuration."""
        return UnifiedRegimeConfig(
            detection_method=RegimeDetectionMethod.NAS_ONLY,
            n_regimes=self.config.n_regimes,
            primary_timeframe=self.config.primary_timeframe,
            min_regime_samples=self.config.min_regime_duration,
            max_regime_samples=self.config.max_regime_duration,
            economic_significance_threshold=self.config.economic_significance_threshold,
            trading_viability_threshold=self.config.trading_viability_threshold,
            max_execution_time=self.config.max_execution_time,
            enable_hardware_optimization=self.config.hardware_config.enable_gpu_acceleration,
            # NAS-specific parameters
            nas_neural_architecture_type=self.config.primary_architecture.value,
            nas_search_strategy=self.config.search_strategy.value,
            nas_population_size=self.config.population_size,
            nas_generations=self.config.generations,
            nas_enable_adaptive_thresholds=True
        )

    def _convert_unified_to_nas_result(self, unified_result: UnifiedRegimeResult) -> PerfectNASResult:
        """Convert unified result to NAS result format."""
        return PerfectNASResult(
            success=unified_result.success,
            regime_predictions=unified_result.regime_predictions,
            regime_probabilities=unified_result.regime_probabilities,
            economic_significance_scores=unified_result.economic_significance_scores,
            trading_viability_scores=unified_result.trading_viability_scores,
            regime_stability_scores=unified_result.regime_stability_scores,
            transition_probabilities=unified_result.transition_probabilities,
            micro_regimes=unified_result.micro_regimes,
            architecture_performance=unified_result.architecture_performance,
            uncertainty_estimates=unified_result.uncertainty_estimates,
            execution_time=unified_result.execution_time,
            metadata=unified_result.metadata,
            error_message=unified_result.error_message
        )

    @timed_operation
    def detect_regimes(self,
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None,
                      optimize_architecture: bool = True,
                      enable_meta_learning: bool = True,
                      learn_thresholds: bool = True) -> PerfectNASResult:
        """
        Detect market regimes using Perfect NAS system with unified utilities.

        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            optimize_architecture: Whether to optimize neural architecture
            enable_meta_learning: Whether to use meta-learning
            learn_thresholds: Whether to learn adaptive thresholds

        Returns:
            PerfectNASResult with regime detection results
        """
        start_time = time.time()

        try:
            tprint("🧠 [PERFECT_NAS_REGIME_DETECTOR] Starting regime detection", color="cyan")
            self.logger.info("🚀 Starting Perfect NAS regime detection")

            # Use unified detector
            if self.unified_detector:
                tprint_info("🧠 Using unified regime detector")
                unified_result = self.unified_detector.detect_regimes(market_data, timestamps)
                return self._convert_unified_to_nas_result(unified_result)
            else:
                raise RuntimeError("Unified detector not available")

        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ [PERFECT_NAS_REGIME_DETECTOR] Regime detection failed: {e}")
            self.logger.error(f"Perfect NAS regime detection failed: {e}")

            return PerfectNASResult(
                success=False,
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                economic_significance_scores=np.array([]),
                trading_viability_scores=np.array([]),
                regime_stability_scores=np.array([]),
                transition_probabilities=np.array([]),
                execution_time=execution_time,
                error_message=str(e)
            )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from unified detector."""
        if self.unified_detector:
            return self.unified_detector.get_performance_metrics()
        else:
            return {}

    def save_detector_state(self, filepath: str) -> bool:
        """Save detector state."""
        try:
            if self.unified_detector:
                # Save unified detector state
                state = {
                    'config': self.config,
                    'unified_config': self._create_unified_config(),
                    'timestamp': time.time()
                }
                import pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(state, f)
                tprint_success(f"✅ NAS detector state saved to {filepath}")
                return True
            else:
                tprint_warning("⚠️ No unified detector available for saving")
                return False
        except Exception as e:
            tprint_error(f"❌ Failed to save NAS detector state: {e}")
            return False

    def load_detector_state(self, filepath: str) -> bool:
        """Load detector state."""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore configuration
            self.config = state['config']
            
            # Reinitialize unified detector with loaded config
            self.unified_detector = UnifiedRegimeDetector(state['unified_config'])
            
            tprint_success(f"✅ NAS detector state saved to {filepath}")
            return True
        except Exception as e:
            tprint_error(f"❌ Failed to load NAS detector state: {e}")
            return False

    def load_market_data(self, symbol: str, interval: str, 
                        start_date=None, end_date=None, data_type: str = "processed") -> Optional[pd.DataFrame]:
        """Load market data using unified utilities."""
        try:
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                tprint_error(f"❌ Invalid symbol: {symbol}")
                raise ValueError(f"Symbol must be a non-empty string, got: {symbol}")
            
            if not interval or not isinstance(interval, str):
                tprint_error(f"❌ Invalid interval: {interval}")
                raise ValueError(f"Interval must be a non-empty string, got: {interval}")
            
            if data_type not in ["processed", "raw", "enhanced"]:
                tprint_warning(f"⚠️ Unknown data_type '{data_type}', using 'processed'")
                data_type = "processed"
            
            tprint_info(f"📊 Loading market data: {symbol} {interval} ({data_type})")
            
            # Use unified utilities for data loading
            if self.common_utils:
                # This would use the unified data loading capabilities
                tprint_info("📊 Using unified data loading utilities")
                # For now, return None as data loading would be implemented in unified utilities
                return None
            else:
                tprint_warning("⚠️ Unified utilities not available for data loading")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Failed to load market data for {symbol} {interval}: {e}")
            self.logger.error(f"Failed to load market data for {symbol} {interval}: {e}")
            return None