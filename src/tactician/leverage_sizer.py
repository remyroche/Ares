"""
Simplified Leverage Sizer for high leverage trading.
Uses ML confidence scores, liquidation risk model, and market health analysis.
"""

import contextlib
from datetime import datetime
import math
from typing import Any

from src.core.decorators import handles_errors
from src.utils.linear_confidence_scaling import LinearConfidenceScaler
from src.utils.logger import system_logger
from src.utils.math_validation import MathValidationError
from src.utils.math_validation import safe_divide
from src.utils.math_validation import safe_log
from src.utils.math_validation import validate_positive
from src.utils.math_validation import validate_range
from src.config.leverage_constants import MIN_LEVERAGE, MAX_LEVERAGE, validate_leverage
from src.utils.validation.unified_framework import (
    safe_divide as unified_safe_divide, safe_log as unified_safe_log,
    validate_positive as unified_validate_positive, validate_range as unified_validate_range,
    MathValidationError as unified_MathValidationError
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
        self.min_leverage: float = leverage_optimization.get('min_leverage', MIN_LEVERAGE)
        self.max_leverage: float = leverage_optimization.get('max_leverage', MAX_LEVERAGE)
        self.leverage_combined_threshold: float = leverage_optimization.get('leverage_combined_threshold', 0.75)
        self.leverage_multiplier: float = leverage_optimization.get('leverage_multiplier', 1.0)
        self.linear_scaler = LinearConfidenceScaler(config)
        self.is_initialized: bool = False
        self.leverage_sizing_history: list[dict[str, Any]] = []

    @handles_errors(
        error_handlers={
            ValueError: (False, 'Invalid leverage sizer configuration'),
            AttributeError: (False, 'Missing required leverage parameters'),
            KeyError: (False, 'Missing configuration keys')
        },
        default_return=False,
        context='leverage sizer initialization'
    )
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
            required_keys = ['min_leverage', 'max_leverage', 'leverage_multiplier']
            for key in required_keys:
                if not hasattr(self, key):
                    self.logger.error(f'Missing required configuration key: {key}')
                    return False
            if self.min_leverage <= 0 or self.min_leverage >= self.max_leverage:
                self.logger.error('Invalid leverage range configuration')
                return False
            if self.leverage_multiplier <= 0:
                self.logger.error('Invalid leverage_multiplier configuration')
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
                self.leverage_combined_threshold = leverage_optimization.get('leverage_combined_threshold', self.leverage_combined_threshold)
                self.leverage_multiplier = leverage_optimization.get('leverage_multiplier', self.leverage_multiplier)
                self.logger.info('✅ Leverage sizer configuration refreshed from step17 results')
        except Exception as e:
            self.logger.exception(f'Error refreshing step17 configuration: {e}')

    @handles_errors(
        error_handlers={
            ValueError: (None, 'Invalid input data for leverage sizing'),
            AttributeError: (None, 'Sizer not properly initialized')
        },
        default_return={},
        context='leverage sizing calculation'
    )
    async def calculate_leverage(self, ml_predictions: dict[str, Any], current_price: float = 0.0, account_balance: float = 1000.0, analyst_confidence: float = 0.5, tactician_confidence: float = 0.5, market_health_analysis: dict[str, Any] | None = None, strategist_risk_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Calculate leverage using ML confidence scores with simplified approach.

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

            # Simplified leverage calculation: ML confidence * multiplier, capped between min/max
            base_leverage = combined_confidence * self.leverage_multiplier
            # Validate and clamp leverage to centralized limits
            final_leverage = validate_leverage(base_leverage)
            final_leverage = max(self.min_leverage, min(self.max_leverage, final_leverage))

            leverage_analysis = {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'account_balance': account_balance,
                'base_leverage': base_leverage,
                'final_leverage': final_leverage,
                'combined_confidence': combined_confidence,
                'intensity': intensity,
                'reliability': reliability,
                'risk_score': risk_score,
                'leverage_multiplier': self.leverage_multiplier,
                'min_leverage': self.min_leverage,
                'max_leverage': self.max_leverage,
                'price_target_confidences': price_target_confidences,
                'adversarial_confidences': adversarial_confidences,
                'market_health_modifiers': market_health_analysis or {},
                'strategist_risk_parameters': strategist_risk_parameters or {},
                'leverage_reason': self._generate_simplified_leverage_reason(final_leverage, combined_confidence, base_leverage)
            }
            self.leverage_sizing_history.append(leverage_analysis)
            if len(self.leverage_sizing_history) > 100:
                self.leverage_sizing_history = self.leverage_sizing_history[-100:]
            self.logger.info(f'✅ Leverage calculated: {final_leverage:.1f}x (ML confidence: {combined_confidence:.3f})')
            return leverage_analysis
        except Exception as e:
            self.logger.exception(f'Error calculating leverage: {e}')
            return {}

    # Simplified leverage calculation - old complex methods removed
    # Now using: leverage = ml_confidence * leverage_multiplier, capped between min/max

    def _generate_simplified_leverage_reason(self, final_leverage: float, combined_confidence: float, base_leverage: float) -> str:
        """Generate reason for simplified leverage sizing decision."""
        try:
            if combined_confidence < self.leverage_combined_threshold:
                return f'Leverage: {final_leverage:.1f}x (minimum due to low ML confidence {combined_confidence:.3f} below threshold {self.leverage_combined_threshold:.3f})'
            elif final_leverage == self.max_leverage:
                return f'Leverage: {final_leverage:.1f}x (maximum due to high ML confidence {combined_confidence:.3f} * multiplier {self.leverage_multiplier:.2f})'
            elif final_leverage == self.min_leverage:
                return f'Leverage: {final_leverage:.1f}x (minimum due to low ML confidence {combined_confidence:.3f} * multiplier {self.leverage_multiplier:.2f})'
            else:
                return f'Leverage: {final_leverage:.1f}x (ML confidence: {combined_confidence:.3f} * multiplier: {self.leverage_multiplier:.2f} = {base_leverage:.1f}x, capped)'
        except Exception as e:
            self.logger.exception(f'Error generating leverage reason: {e}')
            return f'Leverage: {final_leverage:.1f}x (Error generating reason)'

    def get_leverage_sizing_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get leverage sizing history."""
        if limit:
            return self.leverage_sizing_history[-limit:]
        return self.leverage_sizing_history.copy()

    @handles_errors(fallback=None)
    async def stop(self) -> None:
        """Stop the leverage sizer."""
        try:
            self.logger.info('Stopping leverage sizer...')
            self.is_initialized = False
            self.logger.info('✅ Leverage sizer stopped successfully')
        except Exception as e:
            self.logger.exception(f'❌ Failed to stop leverage sizer: {e}')

    @handles_errors(fallback=None)
    async def cleanup(self) -> None:
        """Cleanup leverage sizer resources."""
        try:
            self.logger.info('Cleaning up leverage sizer...')
            await self.stop()
            self.leverage_sizing_history.clear()
            self.logger.info('✅ Leverage sizer cleanup completed')
        except Exception as e:
            self.logger.exception(f'Error cleaning up leverage sizer: {e}')

@handles_errors(fallback=None)
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
