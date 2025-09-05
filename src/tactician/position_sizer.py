"""
Simplified Position Sizer for high leverage trading.
Uses ML confidence scores and Kelly criterion for position sizing.
"""
import contextlib
from datetime import datetime
from typing import Any
from kelly_criterion_fix import calculate_correct_kelly_position_size
from .utils.confidence import normalize_dual_confidence
from .utils.linear_confidence_scaling import LinearConfidenceScaler
from .utils.logger import system_logger
from .utils.warning_symbols import error, initialization_error, missing
from .core.domain.decorators import validate_data_quality

def core_handles_errors(*_args, **kwargs) -> None:
    fallback = kwargs.get('default_return', kwargs.get('fallback', None))
    return _handles_errors(fallback=fallback)

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

    @core_handles_errors(fallback=False)
    async def initialize(self) -> bool:
        """Initialize the position sizer."""
        self.logger.info('Initializing position sizer...')
        if not self._validate_configuration():
            return False
        self.is_initialized = True
        self.logger.info('✅ Position sizer initialized successfully')
        return True

    @core_handles_errors(fallback=None)
    def _validate_configuration(self) -> bool:
        """Validate position sizer configuration."""
        try:
            required_keys = ['kelly_multiplier', 'max_position_size', 'min_position_size']
            for key in required_keys:
                if key not in self.sizing_config:
                    self.print(missing(f'Missing required configuration key: {key}'))
                    return False
            if self.max_position_size <= self.min_position_size:
                self.logger.error('max_position_size must be greater than min_position_size')
                return False
            if self.kelly_multiplier <= 0 or self.kelly_multiplier > 1:
                self.print(error('kelly_multiplier must be between 0 and 1'))
                return False
            return True
        except Exception as e:
            self.print(error(f'Error validating configuration: {e}'))
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

    @validate_data_quality(required_columns=None, min_rows=1, max_null_ratio=0.0, check_duplicates=False, check_timestamps=False, context='position sizing calculation input validation')
    @core_handles_errors(fallback=None)
    async def calculate_position_size(self, ml_predictions: dict[str, Any], current_price: float=0.0, account_balance: float=1000.0, analyst_confidence: float=0.5, tactician_confidence: float=0.5, market_health_analysis: dict[str, Any] | None=None, strategist_risk_parameters: dict[str, Any] | None=None) -> dict[str, Any] | None:
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
            self.print(initialization_error('Position sizer not initialized'))
            return None
        self.logger.info('Calculating position size using ML intelligence...')
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
            confidence_multiplier = self.linear_scaler.calculate_position_size_multiplier(confidence=combined_confidence, intensity=intensity, reliability=reliability, risk_score=risk_score)
            confidence_adjusted_size = base_position_size * confidence_multiplier
            final_position_size = self._apply_position_size_modifiers(confidence_adjusted_size, market_health_analysis=market_health_analysis, strategist_risk_parameters=strategist_risk_parameters, analyst_confidence=analyst_confidence, tactician_confidence=tactician_confidence)
            sizing_analysis = {'timestamp': datetime.now(), 'current_price': current_price, 'account_balance': account_balance, 'kelly_position_size': kelly_position_size, 'ml_position_size': ml_position_size, 'base_position_size': base_position_size, 'confidence_adjusted_size': confidence_adjusted_size, 'final_position_size': final_position_size, 'combined_confidence': combined_confidence, 'intensity': intensity, 'reliability': reliability, 'risk_score': risk_score, 'confidence_multiplier': confidence_multiplier, 'linear_scaling_enabled': True, 'price_target_confidences': price_target_confidences, 'adversarial_confidences': adversarial_confidences, 'directional_confidence': directional_confidence, 'market_health_modifiers': market_health_analysis or {}, 'strategist_risk_parameters': strategist_risk_parameters or {}, 'sizing_reason': self._generate_sizing_reason(final_position_size, kelly_position_size, ml_position_size, price_target_confidences, adversarial_confidences, combined_confidence)}
            self.position_sizing_history.append(sizing_analysis)
            if len(self.position_sizing_history) > 100:
                self.position_sizing_history = self.position_sizing_history[-100:]
            self.logger.info(f'✅ Position size calculated: {final_position_size:.4f}')
            return sizing_analysis
        except Exception as e:
            self.print(error(f'Error calculating position size: {e}'))
            return None

    def _calculate_kelly_position_size(self, price_target_confidences: dict[str, float], adversarial_confidences: dict[str, float]) -> float:
        """Calculate position size using Kelly criterion based on ML confidence scores."""
        try:
            kelly_position_size = calculate_correct_kelly_position_size(price_target_confidences=price_target_confidences, adversarial_confidences=adversarial_confidences, kelly_multiplier=self.kelly_multiplier, min_position_size=self.min_position_size, max_position_size=self.max_position_size)
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
                closest_level = min(price_target_confidences.keys(), key=lambda x: abs(float(x.replace('%', '')) - level))
                confidence = price_target_confidences.get(closest_level, 0.5)
                confidences.append(confidence)
            avg_confidence = sum(confidences) / len(confidences)
            adverse_risks = []
            for level in target_levels:
                closest_level = min(adversarial_confidences.keys(), key=lambda x: abs(float(x.replace('%', '')) - level))
                risk = adversarial_confidences.get(closest_level, 0.3)
                adverse_risks.append(risk)
            avg_adverse_risk = sum(adverse_risks) / len(adverse_risks)
            confidence_factor = avg_confidence / self.confidence_threshold
            risk_factor = 1.0 - avg_adverse_risk
            base_position_size = self.min_position_size + (self.max_position_size - self.min_position_size) * confidence_factor * risk_factor
            return max(self.min_position_size, min(self.max_position_size, base_position_size))
        except (ValueError, TypeError, KeyError) as e:
            self.logger.exception(f'Error calculating ML position size: {e}')
            return self.min_position_size
        except ZeroDivisionError as e:
            self.logger.exception(f'Division by zero in ML position calculation: {e}')
            return self.min_position_size

    def _calculate_weighted_position_size(self, kelly_position_size: float, ml_position_size: float) -> float:
        """Calculate weighted position size using logarithmic computations to prevent multiplicative compounding."""
        try:
            import math
            epsilon = 1e-08
            log_kelly = math.log(kelly_position_size + epsilon)
            log_ml = math.log(ml_position_size + epsilon)
            weighted_log = (1 - self.ml_weight) * log_kelly + self.ml_weight * log_ml
            weighted_size = math.exp(weighted_log)
            return max(self.min_position_size, min(self.max_position_size, weighted_size))
        except Exception as e:
            self.print(error(f'Error calculating weighted position size: {e}'))
            return max(self.min_position_size, min(self.max_position_size, kelly_position_size))

    def _apply_position_size_modifiers(self, base_size: float, *, market_health_analysis: dict[str, Any] | None, strategist_risk_parameters: dict[str, Any] | None, analyst_confidence: float, tactician_confidence: float) -> float:
        """Adjust position size using logarithmic computations to prevent multiplicative compounding."""
        try:
            import math
import numpy as np

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
            self.print(error(f'Error applying size modifiers: {e}'))
            return max(self.min_position_size, min(self.max_position_size, base_size))

    def _generate_sizing_reason(self, final_position_size: float, kelly_position_size: float, ml_position_size: float, price_target_confidences: dict[str, float], adversarial_confidences: dict[str, float], combined_confidence: float=0.5) -> str:
        """Generate reason for position sizing decision."""
        try:
            key_levels = [0.25, 0.5, 0.75, 1.0]
            confidences = []
            risks = []
            for level in key_levels:
                closest_confidence = min(price_target_confidences.keys(), key=lambda x: abs(float(x.replace('%', '')) - level))
                closest_risk = min(adversarial_confidences.keys(), key=lambda x: abs(float(x.replace('%', '')) - level))
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
            self.print(error(f'Error generating sizing reason: {e}'))
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
            self.print(error(f'Error getting historical performance: {e}'))
            return (0.5, 1.5)

    def get_position_sizing_history(self, limit: int | None=None) -> list[dict[str, Any]]:
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
            self.print(error(f'Error stopping position sizer: {e}'))
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

@core_handles_errors(fallback=None)
async def setup_position_sizer(config: dict[str, Any] | None=None) -> PositionSizer | None:
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
        system_logger.exception(error(f'Error setting up position sizer: {e}'))
        return None