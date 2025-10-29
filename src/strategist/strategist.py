

from typing import Any, TYPE_CHECKING

# Note: compat module has been refactored, using enhanced_error_handler instead
from ..utils.enhanced_error_handler import handle_errors_with_tracking
from ..utils.logger import system_logger
from ..core.error_classes import ValidationError
from ..core.decorators import handles_errors
from ..utils.compat import handle_specific_errors
# Performance monitoring
from src.utils.performance_utils import PerformanceMonitor, global_monitor
from src.utils.unified_cache import cached
# Live trading validation
import pandas as pd

from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success
)

"""
Strategist module for regime detection.

This module provides the Strategist class which is responsible for:
- Regime Detection: Load ML models from market_analysis/ and generate regime predictions
- Regime Distribution: Send regime predictions to Analyst & Tactician
"""

# Import Pydantic models and utilities
from .config import StrategistConfig

from .utils import (
    ValidationError,
    log_error,
    validate_data_sufficiency,
    validate_required_columns,
)

if TYPE_CHECKING:
    from src.analyst.analyst import Analyst
    from src.tactician.tactician import Tactician

class Strategist:
    """
    Strategist component responsible for:
    - Regime Detection: Load ML models from market_analysis/ and generate regime predictions
    - Regime Distribution: Send regime predictions to Analyst & Tactician
    
    Note: Does NOT:
    - Generate trading strategies (removed)
    - Calculate market indicators (removed)
    - Apply risk management (done by Supervisor)
    - Provide regime-specific strategy adjustments (removed)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize strategist with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("Strategist")

        # Parse configuration using Pydantic
        strategist_config_dict = self.config.get("strategist", {})
        self.strategist_config = StrategistConfig(**strategist_config_dict)

        # Regime detector (loads models from market_analysis/)
        self.regime_detector: Any = None
        
        # Component references (will be set during initialization)
        self.analyst: Analyst | None = None
        self.tactician: Tactician | None = None
        
        # Signal pipeline reference (for sending regime predictions)
        self.signal_pipeline: Any = None
        
        # Strategist state
        self.is_running: bool = False
        self.current_regime_prediction: dict[str, Any] | None = None

        # Performance monitoring for live trading
        self.performance_monitor: PerformanceMonitor | None = None
        self.global_monitor = global_monitor

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid strategist configuration"),
            AttributeError: (False, "Missing required strategist parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return = False,
        context="strategist initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize strategist with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Strategist...")

            # Configuration is already validated by Pydantic
            self.logger.info("✅ Configuration validated successfully")

            # Initialize regime detector (loads models from market_analysis/)
            await self._initialize_regime_detector()

            # Initialize performance monitoring
            await self._initialize_performance_monitoring()

            self.logger.info("✅ Strategist initialized successfully")
            return True

        except Exception as e:
            log_error(self.logger, "❌ Strategist initialization failed", e)
            return False

    async def _initialize_regime_detector(self) -> None:
        """Initialize regime detector (loads models from market_analysis/)."""
        try:
            from .regime_detector import RegimeDetector
            
            # Get regime detector config
            regime_config = self.config.get("strategist", {}).get("regime_detector", {})
            regime_config.setdefault("models_directory", "artifacts/regime_models")
            
            self.regime_detector = RegimeDetector(regime_config)
            await self.regime_detector.initialize()
            
            self.logger.info("✅ Regime detector initialized")
            tprint("✅ Regime detector initialized")

        except Exception as e:
            error_msg = f"Failed to initialize regime detector: {e}"
            log_error(self.logger, error_msg, e)
            raise RuntimeError(error_msg) from e

    @handle_specific_errors(
        error_handlers={
            ValidationError: (None, "Invalid market data for regime detection"),
            Exception: (None, "Unexpected error in regime detection"),
        },
        default_return = None,
        context="regime detection",
    )
    @cached(ttl=60, key_func=lambda self, market_data: f"regime_{hash(str(market_data.tail(10).values.tolist()))}")
    @global_monitor.track_function
    async def predict_regime(
        self,
        market_data: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """
        Predict market regime using ML models from market_analysis/.
        
        This is the ONLY responsibility of Strategist now.
        
        Args:
            market_data: Market data DataFrame (OHLCV format)

        Returns:
            Regime prediction dictionary or None if failed
        """
        try:
            # Start performance monitoring
            if self.performance_monitor:
                self.performance_monitor.start_timer("regime_detection")

            # Fast fail if regime detector not initialized
            if self.regime_detector is None or not self.regime_detector.is_initialized:
                error_msg = "Regime detector not initialized"
                self.logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)

            self.logger.info("🔍 Detecting market regime...")
            tprint("🔍 Detecting market regime...")

            # Predict regime using loaded models
            regime_prediction = await self.regime_detector.predict_regime(market_data)
            
            if regime_prediction is None:
                error_msg = "Regime detector returned None prediction"
                self.logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)

            # Store current prediction
            self.current_regime_prediction = regime_prediction

            # Send regime prediction to Analyst & Tactician via signal pipeline
            if self.signal_pipeline:
                self.signal_pipeline.regime_detector = self.regime_detector
                self.logger.info("✅ Regime prediction sent to signal pipeline")

            # End performance monitoring
            if self.performance_monitor:
                execution_time = self.performance_monitor.end_timer("regime_detection")
                self.logger.info(f"Regime detection completed in {execution_time:.3f}s")

            self.logger.info(
                f"✅ Regime detected: {regime_prediction.get('primary_regime', 'UNKNOWN')} "
                f"(confidence: {regime_prediction.get('confidence', 0.0):.3f})"
            )
            tprint(
                f"✅ Regime detected: regime_{regime_prediction.get('primary_regime', 0)} "
                f"(confidence: {regime_prediction.get('confidence', 0.0):.3f})"
            )
            
            return regime_prediction

        except Exception as e:
            error_msg = f"Regime detection failed: {e}"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            log_error(self.logger, "Regime detection failed", e)

            # End performance monitoring even on error
            if self.performance_monitor:
                self.performance_monitor.end_timer("regime_detection")

            raise RuntimeError(error_msg) from e

    def _validate_market_data(self, market_data: pd.DataFrame) -> None:
        """
        Validate market data for regime detection.

        Raises:
            ValidationError: If validation fails
        """
        required_columns = ["close", "volume", "timestamp"]
        validate_required_columns(market_data, required_columns)
        validate_data_sufficiency(market_data, min_rows=100)

    @handles_errors(Exception, fallback = False)
    async def _initialize_performance_monitoring(self) -> bool:
        """Initialize performance monitoring."""
        try:
            self.logger.info("Initializing performance monitoring...")

            # Initialize Performance Monitor
            self.performance_monitor = PerformanceMonitor()
            self.logger.info("✅ Performance Monitor initialized")

            # Enable global monitoring
            self.global_monitor.enable()
            self.logger.info("✅ Global monitoring enabled")

            return True
        except Exception as e:
            self.logger.error(f"❌ Error initializing performance monitoring: {e}")
            return False

    @handle_errors_with_tracking(
        context="strategist cleanup",
        log_level="INFO",
        print_errors=True
    )
    async def stop(self) -> bool:
        """Stop the strategist component."""
        try:
            self.logger.info("Stopping Strategist...")
            tprint("Stopping Strategist...")
            self.is_running = False

            # Clean up regime detector
            if self.regime_detector:
                try:
                    await self.regime_detector.stop()
                    self.logger.info("✅ Regime detector stopped")
                    tprint("✅ Regime detector stopped")
                except Exception as e:
                    self.logger.error(f"❌ Error stopping regime detector: {e}")
                    tprint(f"❌ Error stopping regime detector: {e}")

            if self.performance_monitor:
                try:
                    self.performance_monitor.stop()
                    self.logger.info("✅ Performance monitor stopped")
                    tprint("✅ Performance monitor stopped")
                except Exception as e:
                    self.logger.error(f"❌ Error stopping performance monitor: {e}")
                    tprint(f"❌ Error stopping performance monitor: {e}")

            self.logger.info("✅ Strategist stopped successfully")
            tprint("✅ Strategist stopped successfully")
            return True

        except Exception as e:
            error_msg = f"❌ Failed to stop Strategist: {e}"
            self.logger.error(error_msg)
            tprint(error_msg)
            log_error(self.logger, "❌ Failed to stop Strategist", e)
            return False
