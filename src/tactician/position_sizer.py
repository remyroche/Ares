"""
Simplified Position Sizer for high leverage trading.
Uses ML confidence scores and Kelly criterion for position sizing.
"""

import contextlib
from datetime import datetime
import math
from typing import Any, Dict

from src.core.decorators import handles_errors
from src.core.domain.decorators import validate_data_quality
# Kelly criterion calculation function
def calculate_correct_kelly_position_size(price_target_confidences: dict[str, float], adversarial_confidences: dict[str, float], kelly_multiplier: float, min_position_size: float, max_position_size: float) -> float:
    """Calculate Kelly criterion position size."""
    try:
        # Simple Kelly calculation based on confidence scores
        avg_confidence = sum(price_target_confidences.values()) / len(price_target_confidences) if price_target_confidences else 0.5
        avg_risk = sum(adversarial_confidences.values()) / len(adversarial_confidences) if adversarial_confidences else 0.3
        
        # Kelly formula: f = (bp - q) / b where b = odds, p = win probability, q = loss probability
        win_prob = avg_confidence
        loss_prob = 1 - win_prob
        odds = 1.0 / max(avg_risk, 0.1)  # Avoid division by zero
        
        kelly_fraction = (odds * win_prob - loss_prob) / odds
        kelly_fraction = max(0, min(1, kelly_fraction))  # Clamp between 0 and 1
        
        position_size = kelly_fraction * kelly_multiplier
        return max(min_position_size, min(max_position_size, position_size))
    except Exception:
        return min_position_size
from src.utils.confidence import normalize_dual_confidence
from src.utils.linear_confidence_scaling import LinearConfidenceScaler
from src.utils.logger import system_logger
from src.utils.math_validation import MathValidationError
from src.utils.math_validation import safe_divide
from src.utils.math_validation import safe_kelly_calculation
from src.utils.math_validation import safe_log
from src.utils.math_validation import validate_positive
from src.utils.math_validation import validate_range
from src.utils.tprint import tprint
# Enhanced error handling and performance monitoring
from src.utils.enhanced_error_handler import handle_errors_with_tracking
from src.utils.performance_utils import PerformanceMonitor, global_monitor
from src.utils.unified_cache import cached
# Live trading utilities
from src.utils.model_manager import ModelManager
# Live trading validation
from src.utils.trading_decorators import validate_trading_inputs
from src.utils.error_handler import handle_trading_errors
from src.utils.validation.unified_framework import (
    safe_divide as unified_safe_divide, safe_log as unified_safe_log, safe_kelly_calculation as unified_safe_kelly_calculation, 
    validate_positive as unified_validate_positive, validate_range as unified_validate_range, MathValidationError as unified_MathValidationError
)

def core_handles_errors(*_args, **kwargs):
    fallback = kwargs.get('default_return', kwargs.get('fallback', None))
    return handles_errors(fallback=fallback)

class PositionSizer:
    """
    Position Sizer component responsible for:
    - Position sizing decisions based on ML confidence scores and Kelly criterion
    - Integration with Strategist for strategy input
    - Position size optimization for high leverage trading

    This is the primary component responsible for position sizing across the system.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild('PositionSizer')
        if not hasattr(self, 'print'):

            def _shim_print(message: str) -> None:
                with contextlib.suppress(Exception):
                    self.logger.error(str(message))
            self.print = _shim_print
        self.sizing_config: dict[str, Any] = self.config.get('position_sizing', {})
        step17_config = self.config.get('step17_optimization', {})
        position_sizing_optimization = step17_config.get('position_sizing', {})
        self.kelly_multiplier: float = position_sizing_optimization.get('kelly_multiplier', 0.25)
        self.max_position_size: float = position_sizing_optimization.get('max_position_size', 0.5)
        self.min_position_size: float = position_sizing_optimization.get('min_position_size', 0.01)
        self.confidence_threshold: float = position_sizing_optimization.get('confidence_threshold', 0.6)
        self.positionsize_combined_threshold: float = position_sizing_optimization.get('positionsize_combined_threshold', 0.7)
        self.ml_weight: float = position_sizing_optimization.get('ml_weight', 0.7)
        self.linear_scaler = LinearConfidenceScaler(config)
        self.is_initialized: bool = False
        self.position_sizing_history: list[dict[str, Any]] = []
        
        # Live trading utilities
        self.model_manager: ModelManager | None = None
        self.selected_model: str | None = None
        self.model_cache: dict[str, Any] = {}
        
        # Performance monitoring for live trading
        self.performance_monitor: PerformanceMonitor | None = None
        self.global_monitor = global_monitor
        self.position_cache: dict[str, Any] = {}

    @core_handles_errors(fallback = False)
    async def initialize(self) -> bool:
        """Initialize the position sizer."""
        self.logger.info('Initializing position sizer...')
        if not self._validate_configuration():
            return False
        
        # Initialize live trading utilities
        await self._initialize_live_trading_utilities()
        
        # Initialize performance monitoring
        await self._initialize_performance_monitoring()
        
        self.is_initialized = True
        self.logger.info('✅ Position sizer initialized successfully')
        return True

    @handle_errors_with_tracking(
        context="position sizer configuration validation",
        log_level="ERROR",
        print_errors=True
    )
    def _validate_configuration(self) -> bool:
        """Validate position sizer configuration."""
        try:
            self.logger.info("Validating position sizer configuration...")
            tprint("Validating position sizer configuration...")
            
            required_keys = ['kelly_multiplier', 'max_position_size', 'min_position_size']
            for key in required_keys:
                if key not in self.sizing_config:
                    error_msg = f'Missing required configuration key: {key}'
                    self.logger.error(error_msg)
                    tprint(f"❌ {error_msg}")
                    tprint(f"❌ {error_msg}")
                    return False
            
            if self.max_position_size <= self.min_position_size:
                error_msg = 'max_position_size must be greater than min_position_size'
                self.logger.error(error_msg)
                tprint(f"❌ {error_msg}")
                return False
            if self.kelly_multiplier <= 0 or self.kelly_multiplier > 1:
                error_msg = 'kelly_multiplier must be between 0 and 1'
                self.logger.error(error_msg)
                tprint(f"❌ {error_msg}")
                tprint(f"❌ {error_msg}")
                return False
            
            self.logger.info("✅ Position sizer configuration validated successfully")
            tprint("✅ Position sizer configuration validated successfully")
            return True
        except Exception as e:
            error_msg = f'Error validating configuration: {e}'
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            tprint(f"❌ {error_msg}")
            return False

    def refresh_step17_configuration(self, step17_results: dict[str, Any]) -> None:
        """
        Refresh configuration from step17 optimization results.
        This method is called automatically when step17 completes.

        Args:
            step17_results: Step17 optimization results
        """
        try:
            if 'position_sizing' in step17_results:
                position_sizing_optimization = step17_results['position_sizing']
                self.kelly_multiplier = position_sizing_optimization.get('kelly_multiplier', self.kelly_multiplier)
                self.max_position_size = position_sizing_optimization.get('max_position_size', self.max_position_size)
                self.min_position_size = position_sizing_optimization.get('min_position_size', self.min_position_size)
                self.confidence_threshold = position_sizing_optimization.get('confidence_threshold', self.confidence_threshold)
                self.logger.info('✅ Position sizer configuration refreshed from step17 results')
        except Exception as e:
            self.logger.exception(f'Error refreshing step17 configuration: {e}')
            raise

    @validate_data_quality(required_columns = None, min_rows = 1, max_null_ratio = 0.0, check_duplicates = False, check_timestamps = False, context='position sizing calculation input validation')
    @core_handles_errors(fallback=None)
    @cached(ttl=300, key_func=lambda self, ml_predictions, current_price, account_balance, analyst_confidence, tactician_confidence, market_health_analysis, strategist_risk_parameters: f"position_size_{ml_predictions.get('combined_confidence', 0.5)}_{current_price}_{account_balance}")
    @global_monitor.track_function
    async def calculate_position_size(self, ml_predictions: dict[str, Any], current_price: float = 0.0, account_balance: float = 1000.0, analyst_confidence: float = 0.5, tactician_confidence: float = 0.5, market_health_analysis: dict[str, Any] | None = None, strategist_risk_parameters: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """
        Calculate position size using ML confidence scores and Kelly criterion.

        Args:
            ml_predictions: ML confidence predictions from ml_confidence_predictor
            current_price: Current market price
            account_balance: Account balance for position sizing
            market_health_analysis: Aggregated indicators from Analyst's MarketHealthAnalyzer
            strategist_risk_parameters: Risk parameters produced by Strategist (fed via Analyst)

        Returns:
            dict[str, Any]: Position sizing analysis
        """
        if not self.is_initialized:
            error_msg = 'Position sizer not initialized'
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            tprint(f"❌ {error_msg}")
            return None
        
        # Start performance monitoring
        if self.performance_monitor:
            self.performance_monitor.start_timer("position_size_calculation")
        
        self.logger.info('Calculating position size using ML intelligence...')
        tprint('Calculating position size using ML intelligence...')
        try:
            combined_confidence = ml_predictions.get('combined_confidence', 0.5)
            price_target_confidences = ml_predictions.get('price_target_confidences', {})
            adversarial_confidences = ml_predictions.get('adversarial_confidences', {})
            directional_confidence = ml_predictions.get('directional_confidence', {})
            intensity = ml_predictions.get('intensity', 1.0)
            reliability = ml_predictions.get('reliability', 1.0)
            risk_score = ml_predictions.get('risk_score', 0.0)
            kelly_position_size = self._calculate_kelly_position_size(price_target_confidences, adversarial_confidences)
            ml_position_size = self._calculate_ml_position_size(price_target_confidences, adversarial_confidences)
            base_position_size = self._calculate_weighted_position_size(kelly_position_size, ml_position_size)
            confidence_multiplier = self.linear_scaler.calculate_position_size_multiplier(confidence = combined_confidence, intensity = intensity, reliability = reliability, risk_score = risk_score)
            confidence_adjusted_size = base_position_size * confidence_multiplier
            final_position_size = self._apply_position_size_modifiers(confidence_adjusted_size, market_health_analysis = market_health_analysis, strategist_risk_parameters = strategist_risk_parameters, analyst_confidence = analyst_confidence, tactician_confidence = tactician_confidence)
            sizing_analysis = {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'account_balance': account_balance,
                'kelly_position_size': kelly_position_size,
                'ml_position_size': ml_position_size,
                'base_position_size': base_position_size,
                'confidence_adjusted_size': confidence_adjusted_size,
                'final_position_size': final_position_size,
                'combined_confidence': combined_confidence,
                'intensity': intensity,
                'reliability': reliability,
                'risk_score': risk_score,
                'confidence_multiplier': confidence_multiplier,
                'linear_scaling_enabled': True,
                'price_target_confidences': price_target_confidences,
                'adversarial_confidences': adversarial_confidences,
                'directional_confidence': directional_confidence,
                'market_health_modifiers': market_health_analysis or {},
                'strategist_risk_parameters': strategist_risk_parameters or {},
                'sizing_reason': self._generate_sizing_reason(
                    final_position_size, kelly_position_size, ml_position_size,
                    price_target_confidences, adversarial_confidences, combined_confidence
                )
            }
            self.position_sizing_history.append(sizing_analysis)
            if len(self.position_sizing_history) > 100:
                self.position_sizing_history = self.position_sizing_history[-100:]
            
            # End performance monitoring
            if self.performance_monitor:
                execution_time = self.performance_monitor.end_timer("position_size_calculation")
                self.logger.info(f"Position size calculation completed in {execution_time:.3f}s")
                tprint(f"Position size calculation completed in {execution_time:.3f}s")
            
            self.logger.info(f'✅ Position size calculated: {final_position_size:.4f}')
            tprint(f'✅ Position size calculated: {final_position_size:.4f}')
            return sizing_analysis
        except Exception as e:
            error_msg = f'Error calculating position size: {e}'
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            tprint(f"❌ {error_msg}")
            
            # End performance monitoring even on error
            if self.performance_monitor:
                self.performance_monitor.end_timer("position_size_calculation")
            
            return None

    def _calculate_kelly_position_size(self, price_target_confidences: dict[str, float], adversarial_confidences: dict[str, float]) -> float:
        """Calculate position size using Kelly criterion based on ML confidence scores."""
        try:
            kelly_position_size = calculate_correct_kelly_position_size(price_target_confidences = price_target_confidences, adversarial_confidences = adversarial_confidences, kelly_multiplier = self.kelly_multiplier, min_position_size = self.min_position_size, max_position_size = self.max_position_size)
            return max(self.min_position_size, min(self.max_position_size, kelly_position_size))
        except (ValueError, TypeError, KeyError) as e:
            self.logger.exception(f'Error calculating Kelly position size: {e}')
            return self.min_position_size
        except ZeroDivisionError as e:
            self.logger.exception(f'Division by zero in Kelly calculation: {e}')
            return self.min_position_size

    def _calculate_ml_position_size(self, price_target_confidences: dict[str, float], adversarial_confidences: dict[str, float]) -> float:
        """Calculate position size based on ML confidence scores."""
        try:
            target_levels = [0.25, 0.5, 0.75, 1.0]
            confidences = []
            for level in target_levels:
                closest_level = min(price_target_confidences.keys(), key = lambda x: abs(float(x.replace('%', '')) - level))
                confidence = price_target_confidences.get(closest_level, 0.5)
                confidences.append(confidence)
            avg_confidence = sum(confidences) / len(confidences)
            adverse_risks = []
            for level in target_levels:
                closest_level = min(adversarial_confidences.keys(), key = lambda x: abs(float(x.replace('%', '')) - level))
                risk = adversarial_confidences.get(closest_level, 0.3)
                adverse_risks.append(risk)
            avg_adverse_risk = sum(adverse_risks) / len(adverse_risks)
            
            # Use safe division to prevent division by zero
            confidence_factor = safe_divide(avg_confidence, self.confidence_threshold, 1.0)
            risk_factor = 1.0 - avg_adverse_risk
            
            # Validate risk factor is positive
            risk_factor = max(0.0, min(1.0, risk_factor))
            
            base_position_size = self.min_position_size + (self.max_position_size - self.min_position_size) * confidence_factor * risk_factor
            return max(self.min_position_size, min(self.max_position_size, base_position_size))
        except (ValueError, TypeError, KeyError) as e:
            self.logger.exception(f'Error calculating ML position size: {e}')
            return self.min_position_size
        except MathValidationError as e:
            self.logger.warning(f'Mathematical validation error in ML position calculation: {e}')
            return self.min_position_size

    def _calculate_weighted_position_size(self, kelly_position_size: float, ml_position_size: float) -> float:
        """Calculate weighted position size using logarithmic computations to prevent multiplicative compounding."""
        try:
            # Use safe logarithm to prevent log of zero or negative numbers
            log_kelly = safe_log(kelly_position_size, default=0.0)
            log_ml = safe_log(ml_position_size, default=0.0)
            
            # Validate weights are in valid range
            ml_weight = validate_range(self.ml_weight, 0.0, 1.0, "ml_weight")
            
            weighted_log = (1 - ml_weight) * log_kelly + ml_weight * log_ml
            weighted_size = math.exp(weighted_log)
            
            # Ensure result is finite
            if not math.isfinite(weighted_size):
                self.logger.warning(f"Non-finite result in weighted position size calculation")
                return max(self.min_position_size, min(self.max_position_size, kelly_position_size))
            
            return max(self.min_position_size, min(self.max_position_size, weighted_size))
        except MathValidationError as e:
            self.logger.warning(f'Mathematical validation error in weighted position size: {e}')
            return max(self.min_position_size, min(self.max_position_size, kelly_position_size))
        except Exception as e:
            tprint(f"❌ Error calculating weighted position size: {e}")
            return max(self.min_position_size, min(self.max_position_size, kelly_position_size))

    def _apply_position_size_modifiers(self, base_size: float, *, market_health_analysis: dict[str, Any] | None, strategist_risk_parameters: dict[str, Any] | None, analyst_confidence: float, tactician_confidence: float) -> float:
        """Adjust position size using logarithmic computations to prevent multiplicative compounding."""
        try:

            epsilon = 1e-08
            log_adjusted = math.log(base_size + epsilon)
            if market_health_analysis:
                vol = market_health_analysis.get('volatility_analysis', {})
                stress = market_health_analysis.get('stress_analysis', {})
                liq = market_health_analysis.get('liquidity_analysis', {})
                current_vol = float(vol.get('current_volatility', 0.02))
                vol_regime = vol.get('volatility_regime', 'normal')
                stress_level = float(stress.get('stress_level', 0.5))
                liquidity_score = float(liq.get('liquidity_score', 0.5))
                if vol_regime in ('high', 'extreme') or current_vol > 0.03:
                    log_adjusted += math.log(0.6)
                elif vol_regime == 'low' and current_vol < 0.015:
                    log_adjusted += math.log(1.1)
                if stress_level >= 0.8:
                    log_adjusted += math.log(0.4)
                elif stress_level >= 0.6:
                    log_adjusted += math.log(0.6)
                elif stress_level >= 0.4:
                    log_adjusted += math.log(0.8)
                if liquidity_score < 0.3:
                    log_adjusted += math.log(0.6)
                elif liquidity_score > 0.7:
                    log_adjusted += math.log(1.05)
            if strategist_risk_parameters:
                max_position_risk = float(strategist_risk_parameters.get('max_position_risk', 0.01))
                if max_position_risk <= 0.005:
                    log_adjusted += math.log(0.8)
            _, normalized = normalize_dual_confidence(analyst_confidence, tactician_confidence)
            conf_scale = 0.8 + 0.4 * normalized
            log_adjusted += math.log(conf_scale)
            adjusted = math.exp(log_adjusted)
            return max(self.min_position_size, min(self.max_position_size, adjusted))
        except Exception as e:
            tprint(f"❌ Error applying size modifiers: {e}")
            return max(self.min_position_size, min(self.max_position_size, base_size))

    def _generate_sizing_reason(self, final_position_size: float, kelly_position_size: float, ml_position_size: float, price_target_confidences: dict[str, float], adversarial_confidences: dict[str, float], combined_confidence: float = 0.5) -> str:
        """Generate reason for position sizing decision."""
        try:
            key_levels = [0.25, 0.5, 0.75, 1.0]
            confidences = []
            risks = []
            for level in key_levels:
                closest_confidence = min(price_target_confidences.keys(), key = lambda x: abs(float(x.replace('%', '')) - level))
                closest_risk = min(adversarial_confidences.keys(), key = lambda x: abs(float(x.replace('%', '')) - level))
                confidences.append(price_target_confidences.get(closest_confidence, 0.5))
                risks.append(adversarial_confidences.get(closest_risk, 0.3))
            avg_confidence = sum(confidences) / len(confidences)
            avg_risk = sum(risks) / len(risks)
            if final_position_size >= self.max_position_size * 0.8:
                return f'Maximum position size due to high combined confidence ({combined_confidence:.2f}) and low risk ({avg_risk:.2f})'
            if final_position_size >= self.max_position_size * 0.5:
                return f'Large position size based on combined confidence ({combined_confidence:.2f}) and Kelly criterion ({kelly_position_size:.3f})'
            if final_position_size >= self.min_position_size * 2:
                return f'Moderate position size with combined confidence ({combined_confidence:.2f}) and balanced risk-reward profile'
            if combined_confidence < self.positionsize_combined_threshold:
                return f'Minimum position size due to low combined confidence ({combined_confidence:.2f}) below threshold ({self.positionsize_combined_threshold:.2f})'
            return f'Conservative position size due to low confidence ({avg_confidence:.2f}) or high risk ({avg_risk:.2f})'
        except Exception as e:
            tprint(f"❌ Error generating sizing reason: {e}")
            return 'Position size calculated using ML intelligence and Kelly criterion'

    def _generate_dual_confidence_sizing_reason(self, final_position_size: float, final_confidence: float, normalized_confidence: float, analyst_confidence: float, tactician_confidence: float, p_avg: float, b_avg: float, fractional_kelly_pct: float) -> str:
        """Generate sizing reason for dual confidence system."""
        try:
            return f'Position size: {final_position_size:.4f} (Final confidence: {final_confidence:.3f}, Normalized: {normalized_confidence:.3f}) Analyst: {analyst_confidence:.2f}, Tactician: {tactician_confidence:.2f} Kelly: p_avg={p_avg:.2f}, b_avg={b_avg:.2f}, frac_kelly={fractional_kelly_pct:.3f}'
        except Exception as e:
            self.logger.exception(f'Error generating dual confidence sizing reason: {e}')
            return f'Position size: {final_position_size:.4f} (Error generating reason)'

    def _get_historical_performance(self) -> tuple[float, float]:
        """Get historical performance data for Kelly criterion calculation."""
        try:
            history = self.position_sizing_history[-500:]
            if not history:
                return (0.5, 1.5)
            pnls = [float(h.get('pnl', 0.0)) for h in history if 'pnl' in h]
            if not pnls:
                return (0.5, 1.5)
            wins = [p for p in pnls if p > 0]
            losses = [-p for p in pnls if p < 0]
            num_trades = len(pnls)
            win_rate = len(wins) / num_trades if num_trades > 0 else 0.5
            avg_win = sum(wins) / len(wins) if wins else 1.0
            avg_loss = sum(losses) / len(losses) if losses else 1.0
            payoff = avg_win / max(avg_loss, 1e-09) if avg_loss else 1.5
            alpha = min(1.0, num_trades / 200.0)
            p_avg = (1 - alpha) * 0.5 + alpha * win_rate
            b_avg = (1 - alpha) * 1.5 + alpha * payoff
            p_avg = max(0.3, min(0.7, p_avg))
            b_avg = max(0.8, min(2.5, b_avg))
            return (p_avg, b_avg)
        except Exception as e:
            tprint(f"❌ Error getting historical performance: {e}")
            return (0.5, 1.5)

    def get_position_sizing_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get position sizing history."""
        if limit:
            return self.position_sizing_history[-limit:]
        return self.position_sizing_history.copy()

    @core_handles_errors(fallback=None)
    async def stop(self) -> None:
        """Stop the position sizer."""
        try:
            self.logger.info('Stopping position sizer...')
            self.is_initialized = False
            self.logger.info('✅ Position sizer stopped successfully')
        except Exception as e:
            tprint(f"❌ Error stopping position sizer: {e}")
            raise

    @core_handles_errors(fallback=None)
    async def cleanup(self) -> None:
        """Cleanup position sizer resources."""
        try:
            self.logger.info('Cleaning up position sizer...')
            await self.stop()
            self.position_sizing_history.clear()
            self.logger.info('✅ Position sizer cleanup completed')
        except Exception as e:
            self.logger.exception(f'Error cleaning up position sizer: {e}')
            raise

    @handle_errors_with_tracking(
        context="live trading utilities initialization",
        log_level="INFO",
        print_errors=True
    )
    async def _initialize_live_trading_utilities(self) -> bool:
        """Initialize live trading utilities."""
        try:
            self.logger.info("Initializing live trading utilities...")
            tprint("Initializing live trading utilities...")
            
            # Initialize Model Manager for model selection and loading
            self.model_manager = ModelManager()
            self.logger.info("✅ Model Manager initialized")
            tprint("✅ Model Manager initialized")
            
            # Load the single position sizing model
            success = await self.load_position_sizing_model()
            if not success:
                self.logger.warning("⚠️ Failed to load position sizing model during initialization")
                tprint("⚠️ Failed to load position sizing model during initialization")
            
            # Initialize caches
            self.model_cache = {}
            self.position_cache = {}
            self.logger.info("✅ Model and position caches initialized")
            tprint("✅ Model and position caches initialized")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Error initializing live trading utilities: {e}")
            tprint(f"❌ Error initializing live trading utilities: {e}")
            return False

    @core_handles_errors(fallback = False)
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

    @validate_trading_inputs(required_fields=["ml_confidence", "current_price", "account_balance"])
    @handle_errors_with_tracking(
        context="live trading position sizing validation",
        log_level="INFO",
        print_errors=True
    )
    async def validate_position_sizing_inputs(self, sizing_inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Validate position sizing inputs for live trading.
        
        Args:
            sizing_inputs: Position sizing inputs to validate
            
        Returns:
            dict: Validation results
        """
        try:
            self.logger.info("Validating position sizing inputs for live trading...")
            tprint("Validating position sizing inputs for live trading...")
            
            validation_results = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Validate ML confidence
            ml_confidence = sizing_inputs.get("ml_confidence", 0.0)
            if not isinstance(ml_confidence, (int, float)) or ml_confidence < 0.0 or ml_confidence > 1.0:
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Invalid ML confidence: {ml_confidence}")
            
            # Validate current price
            current_price = sizing_inputs.get("current_price", 0.0)
            if not isinstance(current_price, (int, float)) or current_price <= 0:
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Invalid current price: {current_price}")
            
            # Validate account balance
            account_balance = sizing_inputs.get("account_balance", 0.0)
            if not isinstance(account_balance, (int, float)) or account_balance <= 0:
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Invalid account balance: {account_balance}")
            
            # Check for reasonable position size bounds
            if validation_results["is_valid"]:
                estimated_position_size = (ml_confidence * account_balance) / current_price
                if estimated_position_size > self.max_position_size * 2:
                    validation_results["warnings"].append(f"Estimated position size ({estimated_position_size:.4f}) is very large")
            
            self.logger.info(f"✅ Position sizing inputs validation completed: {'PASS' if validation_results['is_valid'] else 'FAIL'}")
            tprint(f"✅ Position sizing inputs validation completed: {'PASS' if validation_results['is_valid'] else 'FAIL'}")
            return validation_results
            
        except Exception as e:
            error_msg = f"Error validating position sizing inputs: {e}"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            return {"error": error_msg}

    @handle_errors_with_tracking(
        context="position sizing model loading",
        log_level="INFO",
        print_errors=True
    )
    async def load_position_sizing_model(self) -> bool:
        """
        Load the single position sizing model trained on various market conditions.
        
        Returns:
            bool: True if model loading successful
        """
        if not self.model_manager:
            error_msg = "Model Manager not available"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            return False
        
        try:
            # Use the single position sizing model trained on various market conditions
            model_name = "tactician_position_sizing_model"
            
            self.logger.info(f"Loading position sizing model for live trading: {model_name}")
            tprint(f"Loading position sizing model for live trading: {model_name}")
            
            # Check if model is available
            available_models = await self.model_manager.list_available_models()
            if model_name not in available_models:
                error_msg = f"Position sizing model {model_name} not available for live trading"
                self.logger.error(error_msg)
                tprint(f"❌ {error_msg}")
                return False
            
            # Load and cache the model
            model = await self.model_manager.load_model(model_name)
            if model:
                self.selected_model = model_name
                self.model_cache[model_name] = model
                self.logger.info(f"✅ Position sizing model loaded and cached: {model_name}")
                tprint(f"✅ Position sizing model loaded and cached: {model_name}")
                return True
            else:
                error_msg = f"Failed to load position sizing model: {model_name}"
                self.logger.error(error_msg)
                tprint(f"❌ {error_msg}")
                return False
            
        except Exception as e:
            error_msg = f"Error loading position sizing model: {e}"
            self.logger.error(error_msg)
            tprint(f"❌ {error_msg}")
            return False

@core_handles_errors(fallback=None)
async def setup_position_sizer(config: dict[str, Any] | None = None) -> PositionSizer | None:
    """
    Setup position sizer.

    Args:
        config: Configuration dictionary

    Returns:
        Optional[PositionSizer]: Initialized position sizer or None
    """
    try:
        if config is None:
            config = {}
        position_sizer = PositionSizer(config)
        if await position_sizer.initialize():
            return position_sizer
        return None
    except Exception as e:
        system_logger.exception(f'Error setting up position sizer: {e}')
        return None