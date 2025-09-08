"""
import contextlib
from datetime import datetime
import math
import math
from typing import Any

from core.decorators import handles_errors
from utils.linear_confidence_scaling import LinearConfidenceScaler
from utils.logger import system_logger
from utils.math_validation import MathValidationError
from utils.math_validation import safe_divide
from utils.math_validation import safe_log
from utils.math_validation import validate_positive
from utils.math_validation import validate_range

from ..utils.logger import system_logger
Simplified Leverage Sizer for high leverage trading.
Uses ML confidence scores, liquidation risk model, and market health analysis.
"""
    safe_divide, safe_log, validate_positive, validate_range, MathValidationError
)

# Use the imported handles_errors decorator directly

class LeverageSizer:
    """
    Simplified leverage sizer that uses ML confidence scores and liquidation risk model
    to set leverage between 10x and 100x.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild('LeverageSizer')
        if not hasattr(self, 'print'):

            def _shim_print(message: str) -> None:
                with contextlib.suppress(Exception):
                    self.logger.error(str(message))
            self.print = _shim_print
        self.leverage_config: dict[str, Any] = self.config.get('leverage_sizing', {})
        step17_config = self.config.get('step17_optimization', {})
        leverage_optimization = step17_config.get('leverage', {})
        self.min_leverage: float = leverage_optimization.get('min_leverage', 10.0)
        self.max_leverage: float = leverage_optimization.get('max_leverage', 100.0)
        self.confidence_threshold: float = leverage_optimization.get('confidence_threshold', 0.6)
        self.liquidation_buffer: float = leverage_optimization.get('liquidation_buffer', 0.05)
        self.leverage_combined_threshold: float = leverage_optimization.get('leverage_combined_threshold', 0.75)
        self.ml_weight: float = leverage_optimization.get('ml_weight', 0.6)
        self.liquidation_weight: float = leverage_optimization.get('liquidation_weight', 0.4)
        self.leverage_multiplier: float = leverage_optimization.get('leverage_multiplier', 1.0)
        self.max_risk_leverage: float = leverage_optimization.get('max_risk_leverage', 50.0)
        self.linear_scaler = LinearConfidenceScaler(config)
        self.is_initialized: bool = False
        self.leverage_sizing_history: list[dict[str, Any]] = []

    @handles_errors(error_handlers={ValueError: (False, 'Invalid leverage sizer configuration'), AttributeError: (False, 'Missing required leverage parameters'), KeyError: (False, 'Missing configuration keys')}, default_return = False, context='leverage sizer initialization')
    async def initialize(self) -> bool:
        """Initialize the leverage sizer."""
        self.logger.info('Initializing leverage sizer...')
        if not self._validate_configuration():
            return False
        self.is_initialized = True
        self.logger.info('✅ Leverage sizer initialized successfully')
        return True

    def _validate_configuration(self) -> bool:
        """Validate leverage sizer configuration."""
        try:
            required_keys = ['min_leverage', 'max_leverage', 'confidence_threshold', 'liquidation_buffer']
            for key in required_keys:
                if not hasattr(self, key):
                    self.logger.error(f'Missing required configuration key: {key}')
                    return False
            if self.min_leverage <= 0 or self.min_leverage >= self.max_leverage:
                self.logger.error('Invalid leverage range configuration')
                return False
            if self.liquidation_buffer <= 0 or self.liquidation_buffer >= 1:
                self.logger.error('Invalid liquidation_buffer configuration')
                return False
            return True
        except Exception as e:
            self.logger.exception(f'Configuration validation failed: {e}')
            return False

    def refresh_step17_configuration(self, step17_results: dict[str, Any]) -> None:
        """
        Refresh configuration from step17 optimization results.
        This method is called automatically when step17 completes.

        Args:
            step17_results: Step17 optimization results
        """
        try:
            if 'leverage' in step17_results:
                leverage_optimization = step17_results['leverage']
                self.min_leverage = leverage_optimization.get('min_leverage', self.min_leverage)
                self.max_leverage = leverage_optimization.get('max_leverage', self.max_leverage)
                self.confidence_threshold = leverage_optimization.get('confidence_threshold', self.confidence_threshold)
                self.liquidation_buffer = leverage_optimization.get('liquidation_buffer', self.liquidation_buffer)
                self.ml_weight = leverage_optimization.get('ml_weight', self.ml_weight)
                self.liquidation_weight = leverage_optimization.get('liquidation_weight', self.liquidation_weight)
                self.leverage_multiplier = leverage_optimization.get('leverage_multiplier', self.leverage_multiplier)
                self.max_risk_leverage = leverage_optimization.get('max_risk_leverage', self.max_risk_leverage)
                self.logger.info('✅ Leverage sizer configuration refreshed from step17 results')
        except Exception as e:
            self.logger.exception(f'Error refreshing step17 configuration: {e}')

    @handles_errors(error_handlers={ValueError: (None, 'Invalid input data for leverage sizing'), AttributeError: (None, 'Sizer not properly initialized')}, default_return={}, context='leverage sizing calculation')
    async def calculate_leverage(self, ml_predictions: dict[str, Any], current_price: float = 0.0, account_balance: float = 1000.0, analyst_confidence: float = 0.5, tactician_confidence: float = 0.5, market_health_analysis: dict[str, Any] | None = None, strategist_risk_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Calculate leverage using ML confidence scores and liquidation risk model.

        Args:
            ml_predictions: ML model predictions
            current_price: Current market price
            account_balance: Account balance
            analyst_confidence: Analyst confidence score
            tactician_confidence: Tactician confidence score
            market_health_analysis: Market health analysis
            strategist_risk_parameters: Strategist risk parameters

        Returns:
            dict[str, Any]: Leverage sizing analysis
        """
        if not self.is_initialized:
            self.logger.error('Leverage sizer not initialized')
            return {}
        try:
            combined_confidence = ml_predictions.get('combined_confidence', 0.5)
            price_target_confidences = ml_predictions.get('price_target_confidences', {})
            adversarial_confidences = ml_predictions.get('adversarial_confidences', {})
            intensity = ml_predictions.get('intensity', 1.0)
            reliability = ml_predictions.get('reliability', 1.0)
            risk_score = ml_predictions.get('risk_score', 0.0)
            ml_leverage = self._calculate_ml_leverage(price_target_confidences, adversarial_confidences)
            liquidation_leverage = self._calculate_liquidation_safe_leverage(current_price, account_balance, market_health_analysis)
            base_leverage = self._calculate_weighted_leverage(ml_leverage, liquidation_leverage)
            confidence_multiplier = self.linear_scaler.calculate_leverage_multiplier(confidence = combined_confidence, intensity = intensity, reliability = reliability, risk_score = risk_score)
            confidence_adjusted_leverage = base_leverage * confidence_multiplier
            final_leverage = self._apply_leverage_modifiers(confidence_adjusted_leverage, market_health_analysis = market_health_analysis, strategist_risk_parameters = strategist_risk_parameters, analyst_confidence = analyst_confidence, tactician_confidence = tactician_confidence)
            leverage_analysis = {'timestamp': datetime.now(), 'current_price': current_price, 'account_balance': account_balance, 'ml_leverage': ml_leverage, 'liquidation_leverage': liquidation_leverage, 'base_leverage': base_leverage, 'confidence_adjusted_leverage': confidence_adjusted_leverage, 'final_leverage': final_leverage, 'combined_confidence': combined_confidence, 'intensity': intensity, 'reliability': reliability, 'risk_score': risk_score, 'confidence_multiplier': confidence_multiplier, 'linear_scaling_enabled': True, 'price_target_confidences': price_target_confidences, 'adversarial_confidences': adversarial_confidences, 'market_health_modifiers': market_health_analysis or {}, 'strategist_risk_parameters': strategist_risk_parameters or {}, 'leverage_reason': self._generate_leverage_reason(final_leverage, ml_leverage, liquidation_leverage, price_target_confidences, adversarial_confidences, combined_confidence)}
            self.leverage_sizing_history.append(leverage_analysis)
            if len(self.leverage_sizing_history) > 100:
                self.leverage_sizing_history = self.leverage_sizing_history[-100:]
            self.logger.info(f'✅ Leverage calculated: {final_leverage:.1f}x')
            return leverage_analysis
        except Exception as e:
            self.logger.exception(f'Error calculating leverage: {e}')
            return {}

    def _calculate_ml_leverage(self, price_target_confidences: dict[str, float], adversarial_confidences: dict[str, float]) -> float:
        """Calculate leverage based on ML confidence scores."""
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
            
            base_leverage = self.min_leverage + (self.max_leverage - self.min_leverage) * confidence_factor * risk_factor
            return max(self.min_leverage, min(self.max_leverage, base_leverage))
        except (ValueError, TypeError, KeyError) as e:
            self.logger.exception(f'Error calculating ML leverage: {e}')
            return self.min_leverage
        except MathValidationError as e:
            self.logger.warning(f'Mathematical validation error in ML leverage calculation: {e}')
            return self.min_leverage

    def _calculate_liquidation_safe_leverage(self, current_price: float, account_balance: float, market_health_analysis: dict[str, Any] | None) -> float:
        """Calculate safe leverage to avoid liquidation."""
        try:
            worst_case_move = 0.1
            if market_health_analysis:
                vol_analysis = market_health_analysis.get('volatility_analysis', {})
                current_vol = float(vol_analysis.get('current_volatility', 0.02))
                if current_vol > 0.03:
                    worst_case_move = min(0.2, current_vol * 2)
                elif current_vol > 0.02:
                    worst_case_move = 0.15
            
            # Use safe division to prevent division by zero
            safe_leverage = safe_divide(1.0 - self.liquidation_buffer, worst_case_move, self.min_leverage)
            return max(self.min_leverage, min(self.max_leverage, safe_leverage))
        except (ValueError, TypeError) as e:
            self.logger.exception(f'Error calculating liquidation safe leverage: {e}')
            return self.min_leverage
        except MathValidationError as e:
            self.logger.warning(f'Mathematical validation error in liquidation safe leverage: {e}')
            return self.min_leverage

    def _calculate_weighted_leverage(self, ml_leverage: float, liquidation_leverage: float) -> float:
        """Calculate weighted leverage using logarithmic computations to prevent multiplicative compounding."""
        try:
            
            # Use safe logarithm to prevent log of zero or negative numbers
            log_ml = safe_log(ml_leverage, default=0.0)
            log_liquidation = safe_log(liquidation_leverage, default=0.0)
            
            # Use safe division for weight normalization
            total_weight = self.ml_weight + self.liquidation_weight
            normalized_ml_weight = safe_divide(self.ml_weight, total_weight, 0.5)
            normalized_liquidation_weight = safe_divide(self.liquidation_weight, total_weight, 0.5)
            
            weighted_log = normalized_ml_weight * log_ml + normalized_liquidation_weight * log_liquidation
            weighted_leverage = math.exp(weighted_log)
            
            # Ensure result is finite
            if not math.isfinite(weighted_leverage):
                self.logger.warning(f"Non-finite result in weighted leverage calculation")
                return self.min_leverage
            
            return max(self.min_leverage, min(self.max_leverage, weighted_leverage))
        except MathValidationError as e:
            self.logger.warning(f'Mathematical validation error in weighted leverage: {e}')
            return self.min_leverage
        except Exception as e:
            self.logger.exception(f'Error calculating weighted leverage: {e}')
            return self.min_leverage

    def _apply_leverage_modifiers(self, base_leverage: float, *, market_health_analysis: dict[str, Any] | None, strategist_risk_parameters: dict[str, Any] | None, analyst_confidence: float, tactician_confidence: float) -> float:
        """Adjust leverage using logarithmic computations to prevent multiplicative compounding."""
        try:

            epsilon = 1e-08
            log_adjusted = math.log(base_leverage + epsilon)
            if market_health_analysis:
                volatility_modifier = market_health_analysis.get('volatility_modifier', 1.0)
                liquidity_modifier = market_health_analysis.get('liquidity_modifier', 1.0)
                stress_modifier = market_health_analysis.get('stress_modifier', 1.0)
                log_adjusted += math.log(volatility_modifier)
                log_adjusted += math.log(liquidity_modifier)
                log_adjusted += math.log(stress_modifier)
            if strategist_risk_parameters:
                risk_modifier = strategist_risk_parameters.get('leverage_modifier', 1.0)
                log_adjusted += math.log(risk_modifier)
            confidence_modifier = (analyst_confidence + tactician_confidence) / 2
            log_adjusted += math.log(confidence_modifier)
            adjusted = math.exp(log_adjusted)
            return max(self.min_leverage, min(self.max_leverage, adjusted))
        except Exception as e:
            self.logger.exception(f'Error applying leverage modifiers: {e}')
            return base_leverage

    def _generate_leverage_reason(self, final_leverage: float, ml_leverage: float, liquidation_leverage: float, price_target_confidences: dict[str, float], adversarial_confidences: dict[str, float], combined_confidence: float = 0.5) -> str:
        """Generate reason for leverage sizing decision."""
        try:
            key_levels = [0.25, 0.5, 0.75, 1.0]
            avg_confidence = 0.0
            avg_risk = 0.0
            for level in key_levels:
                closest_level = min(price_target_confidences.keys(), key = lambda x: abs(float(x.replace('%', '')) - level))
                confidence = price_target_confidences.get(closest_level, 0.5)
                risk = adversarial_confidences.get(closest_level, 0.3)
                avg_confidence += confidence
                avg_risk += risk
            avg_confidence /= len(key_levels)
            avg_risk /= len(key_levels)
            if combined_confidence < self.leverage_combined_threshold:
                return f'Leverage: {final_leverage:.1f}x (minimum due to low combined confidence {combined_confidence:.2f} below threshold {self.leverage_combined_threshold:.2f})'
            return f'Leverage: {final_leverage:.1f}x (ML: {ml_leverage:.1f}x, Liquidation: {liquidation_leverage:.1f}x, Combined Confidence: {combined_confidence:.3f}, Risk: {avg_risk:.3f})'
        except Exception as e:
            self.logger.exception(f'Error generating leverage reason: {e}')
            return f'Leverage: {final_leverage:.1f}x (Error generating reason)'

    def get_leverage_sizing_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get leverage sizing history."""
        if limit:
            return self.leverage_sizing_history[-limit:]
        return self.leverage_sizing_history.copy()

    @handles_errors(fallback = None)
    async def stop(self) -> None:
        """Stop the leverage sizer."""
        try:
            self.logger.info('Stopping leverage sizer...')
            self.is_initialized = False
            self.logger.info('✅ Leverage sizer stopped successfully')
        except Exception as e:
            self.logger.exception(f'❌ Failed to stop leverage sizer: {e}')

    @handles_errors(fallback = None)
    async def cleanup(self) -> None:
        """Cleanup leverage sizer resources."""
        try:
            self.logger.info('Cleaning up leverage sizer...')
            await self.stop()
            self.leverage_sizing_history.clear()
            self.logger.info('✅ Leverage sizer cleanup completed')
        except Exception as e:
            self.logger.exception(f'Error cleaning up leverage sizer: {e}')

@handles_errors(fallback = None)
async def setup_leverage_sizer(config: dict[str, Any] | None = None) -> LeverageSizer | None:
    """
    Setup and return a configured LeverageSizer instance.

    Args:
        config: Configuration dictionary

    Returns:
        LeverageSizer: Configured leverage sizer instance
    """
    try:
        if config is None:
            config = {}
        leverage_sizer = LeverageSizer(config)
        if await leverage_sizer.initialize():
            return leverage_sizer
        return None
    except Exception as e:
        system_logger.exception(f'Failed to setup leverage sizer: {e}')
        return None