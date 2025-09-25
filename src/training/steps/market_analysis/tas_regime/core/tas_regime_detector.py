"""
Tree-Driven Advanced Statistics (TAS) Regime Detector - Clean Version

This module implements the TAS regime detection system using unified utilities.
All legacy code has been removed and the system now relies entirely on
the unified regime detection framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
import time
from dataclasses import dataclass

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import unified utilities
try:
    from src.utils.nas_tas import (
        UnifiedRegimeDetector, UnifiedRegimeConfig, UnifiedRegimeResult,
        RegimeDetectionMethod
    )
    from src.utils.common_operations import (
        CommonUtilities, memory_checkpoint, gpu_context
    )
    UNIFIED_UTILITIES_AVAILABLE = True
except ImportError:
    UNIFIED_UTILITIES_AVAILABLE = False

# Import TAS configuration
from .tas_regime_config import TASRegimeConfig

# Import TAS result
try:
    from .tas_regime_result import TASRegimeResult
except ImportError:
    # Fallback result class if not available
    @dataclass
    class TASRegimeResult:
        success: bool
        regime_predictions: np.ndarray
        regime_probabilities: np.ndarray
        economic_significance_scores: np.ndarray
        trading_viability_scores: np.ndarray
        regime_stability_scores: np.ndarray
        transition_probabilities: np.ndarray
        micro_regimes: Optional[Dict[str, Any]] = None
        uncertainty_estimates: Optional[np.ndarray] = None
        execution_time: float = 0.0
        metadata: Dict[str, Any] = None
        error_message: Optional[str] = None

logger = logging.getLogger(__name__)

class TASRegimeDetector:
    """
    TAS Regime Detector using unified utilities.
    
    This is a clean version that relies entirely on the unified regime detection
    system. All legacy code has been removed.
    """

    def __init__(self, config: TASRegimeConfig):
        """Initialize TAS Regime Detector."""
        tprint_info("🚀 Initializing TAS Regime Detector (Clean Version)")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize unified utilities if available
        if UNIFIED_UTILITIES_AVAILABLE:
            tprint_info("🔧 Initializing unified utilities...")
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
            tprint_error("❌ Unified utilities not available - TAS detector requires unified system")
            raise ImportError("Unified utilities are required for TAS regime detection")

        tprint_success("✅ TAS Regime Detector initialized")
        self.logger.info("✅ TAS Regime Detector initialized")

    def _create_unified_config(self) -> UnifiedRegimeConfig:
        """Create unified configuration from TAS configuration."""
        return UnifiedRegimeConfig(
            detection_method=RegimeDetectionMethod.TAS_ONLY,
            n_regimes=self.config.n_regimes,
            primary_timeframe=self.config.primary_timeframe,
            min_regime_samples=self.config.min_regime_samples,
            max_regime_samples=self.config.max_regime_samples,
            economic_significance_threshold=self.config.economic_significance_threshold,
            trading_viability_threshold=self.config.trading_viability_threshold,
            max_execution_time=self.config.max_execution_time,
            enable_hardware_optimization=self.config.enable_hardware_optimization,
            # TAS-specific parameters
            tas_tree_depth=self.config.tree_depth,
            tas_n_estimators=self.config.n_estimators,
            tas_enable_statistical_methods=self.config.enable_statistical_methods
        )

    def _convert_unified_to_tas_result(self, unified_result: UnifiedRegimeResult) -> TASRegimeResult:
        """Convert unified result to TAS result format."""
        return TASRegimeResult(
            success=unified_result.success,
            regime_predictions=unified_result.regime_predictions,
            regime_probabilities=unified_result.regime_probabilities,
            economic_significance_scores=unified_result.economic_significance_scores,
            trading_viability_scores=unified_result.trading_viability_scores,
            regime_stability_scores=unified_result.regime_stability_scores,
            transition_probabilities=unified_result.transition_probabilities,
            micro_regimes=unified_result.micro_regimes,
            uncertainty_estimates=unified_result.uncertainty_estimates,
            execution_time=unified_result.execution_time,
            metadata=unified_result.metadata,
            error_message=unified_result.error_message
        )

    def detect_regimes(self,
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None,
                      optimize_performance: bool = True,
                      enable_clvsa_enhancement: bool = True) -> TASRegimeResult:
        """
        Detect market regimes using TAS system with unified utilities.

        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            optimize_performance: Whether to use hardware optimization
            enable_clvsa_enhancement: Whether to use CLVSA enhancement

        Returns:
            TASRegimeResult with regime detection results
        """
        start_time = time.time()

        try:
            self.logger.info("🚀 Starting TAS regime detection")

            # Use unified detector
            if self.unified_detector:
                tprint_info("🧠 Using unified regime detector")
                unified_result = self.unified_detector.detect_regimes(market_data, timestamps)
                return self._convert_unified_to_tas_result(unified_result)
            else:
                raise RuntimeError("Unified detector not available")

        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ TAS regime detection failed: {e}")
            self.logger.error(f"TAS regime detection failed: {e}")

            return TASRegimeResult(
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
                tprint_success(f"✅ TAS detector state saved to {filepath}")
                return True
            else:
                tprint_warning("⚠️ No unified detector available for saving")
                return False
        except Exception as e:
            tprint_error(f"❌ Failed to save TAS detector state: {e}")
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
            
            tprint_success(f"✅ TAS detector state loaded from {filepath}")
            return True
        except Exception as e:
            tprint_error(f"❌ Failed to load TAS detector state: {e}")
            return False

    @contextmanager
    def _hardware_optimization_context(self):
        """Context manager for hardware optimization."""
        # Use memory checkpoint if available
        if self.common_utils:
            try:
                with memory_checkpoint("tas_regime_detection"):
                    yield
            except Exception:
                yield
        else:
            yield