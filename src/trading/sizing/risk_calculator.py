"""
Risk Calculator

Simplified risk calculation for position sizing.
Basic risk metrics without complex regime-based calculations.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import math

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from ..config.trading_config import TradingConfig

logger = system_logger.getChild('RiskCalculator')

@dataclass
class RiskMetrics:
    """Risk metrics result."""
    position_risk: float
    portfolio_risk: float
    max_loss: float
    risk_reward_ratio: float
    volatility_risk: float
    metadata: Dict[str, Any]

class RiskCalculator:
    """
    Simplified risk calculator for position sizing.

    Provides basic risk metrics:
    - Position risk calculation
    - Portfolio risk assessment
    - Maximum loss estimation
    - Risk-reward ratio calculation
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logger.getChild('RiskCalculator')

        # Risk parameters - read from config with defaults
        self.max_portfolio_risk: float = getattr(config, 'max_portfolio_risk', 0.02)  # 2% max portfolio risk
        self.max_position_risk: float = 0.01  # 1% max position risk per trade
        self.default_volatility: float = 0.02  # 2% default volatility

        # State management
        self.is_initialized: bool = False

    @handles_errors
    async def initialize(self) -> bool:
        """Initialize risk calculator."""
        try:
            self.logger.info("Initializing Risk Calculator...")

            # Validate configuration
            if not self._validate_configuration():
                return False

            self.is_initialized = True
            self.logger.info("✅ Risk Calculator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Risk Calculator: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate risk calculator configuration."""
        try:
            if self.max_portfolio_risk <= 0 or self.max_portfolio_risk > 1:
                self.logger.error("Invalid max_portfolio_risk configuration")
                return False
            if self.max_position_risk <= 0 or self.max_position_risk > 1:
                self.logger.error("Invalid max_position_risk configuration")
                return False
            if self.default_volatility <= 0 or self.default_volatility > 1:
                self.logger.error("Invalid default_volatility configuration")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def _validate_inputs(
        self,
        position_size: float,
        current_price: float,
        account_balance: float,
        volatility: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ) -> None:
        """Validate input parameters for risk calculations."""
        import math
        
        if not math.isfinite(position_size) or position_size <= 0:
            raise ValueError(f"position_size must be a positive finite number, got {position_size}")
        if not math.isfinite(current_price) or current_price <= 0:
            raise ValueError(f"current_price must be a positive finite number, got {current_price}")
        if not math.isfinite(account_balance) or account_balance <= 0:
            raise ValueError(f"account_balance must be a positive finite number, got {account_balance}")
        if volatility is not None and (not math.isfinite(volatility) or volatility < 0 or volatility > 1):
            raise ValueError(f"volatility must be between 0 and 1, got {volatility}")
        if stop_loss_price is not None and (not math.isfinite(stop_loss_price) or stop_loss_price <= 0):
            raise ValueError(f"stop_loss_price must be a positive finite number, got {stop_loss_price}")
        if take_profit_price is not None and (not math.isfinite(take_profit_price) or take_profit_price <= 0):
            raise ValueError(f"take_profit_price must be a positive finite number, got {take_profit_price}")

    @handles_errors
    async def calculate_risk_metrics(
        self,
        position_size: float,
        current_price: float,
        account_balance: float,
        volatility: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        leverage: float = 1.0
    ) -> RiskMetrics:
        """
        Calculate risk metrics for a position.

        Args:
            position_size: Position size in units
            current_price: Current market price
            account_balance: Account balance
            volatility: Market volatility (optional)
            stop_loss_price: Stop loss price (optional)
            take_profit_price: Take profit price (optional)
            leverage: Leverage multiplier (default 1.0)

        Returns:
            RiskMetrics: Risk metrics for the position
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Risk Calculator not initialized")

            # Validate inputs
            self._validate_inputs(position_size, current_price, account_balance, volatility, stop_loss_price, take_profit_price)

            # Use default volatility if not provided
            if volatility is None:
                volatility = self.default_volatility

            # Calculate position value
            position_value = position_size * current_price

            # Calculate position exposure (as fraction of account balance)
            position_exposure = position_value / account_balance if account_balance > 0 else 0.0

            # Calculate actual position risk (based on stop loss distance)
            if stop_loss_price:
                # Real risk: potential loss as fraction of account balance
                stop_loss_distance_pct = abs(current_price - stop_loss_price) / current_price if current_price > 0 else 0.0
                position_risk = position_exposure * stop_loss_distance_pct * leverage
            else:
                # Use volatility-based stop loss
                position_risk = position_exposure * volatility * leverage

            # Calculate portfolio risk (volatility-adjusted exposure)
            portfolio_risk = position_exposure * volatility * leverage

            # Calculate maximum loss in account currency
            if stop_loss_price:
                max_loss = abs(position_size * (current_price - stop_loss_price))
            else:
                # Use volatility-based stop loss
                stop_loss_distance = current_price * volatility
                max_loss = position_size * stop_loss_distance

            # Calculate risk-reward ratio
            risk_reward_ratio = 0.0
            if take_profit_price and stop_loss_price:
                potential_profit = abs(position_size * (take_profit_price - current_price))
                potential_loss = abs(position_size * (current_price - stop_loss_price))
                if potential_loss > 0:
                    risk_reward_ratio = potential_profit / potential_loss

            # Calculate volatility-adjusted risk (VaR-like metric)
            # This represents the potential loss at the volatility level
            volatility_risk = position_exposure * volatility * leverage

            # Create result
            result = RiskMetrics(
                position_risk=position_risk,
                portfolio_risk=portfolio_risk,
                max_loss=max_loss,
                risk_reward_ratio=risk_reward_ratio,
                volatility_risk=volatility_risk,
                metadata={
                    'position_value': position_value,
                    'position_exposure': position_exposure,
                    'volatility': volatility,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'leverage': leverage,
                    'max_portfolio_risk': self.max_portfolio_risk,
                    'max_position_risk': self.max_position_risk
                }
            )

            self.logger.debug(f"Risk metrics calculated: position_risk={position_risk:.4f}, portfolio_risk={portfolio_risk:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Risk metrics calculation failed: {e}")
            raise

    @handles_errors
    async def validate_position_risk(
        self,
        position_size: float,
        current_price: float,
        account_balance: float,
        volatility: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        leverage: float = 1.0
    ) -> Dict[str, Any]:
        """
        Validate if position risk is within acceptable limits.

        Args:
            position_size: Position size in units
            current_price: Current market price
            account_balance: Account balance
            volatility: Market volatility (optional)

        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            # Calculate risk metrics
            risk_metrics = await self.calculate_risk_metrics(
                position_size, current_price, account_balance, volatility, stop_loss_price, None, leverage
            )

            # Validate position risk
            position_risk_valid = risk_metrics.position_risk <= self.max_position_risk

            # Validate portfolio risk
            portfolio_risk_valid = risk_metrics.portfolio_risk <= self.max_portfolio_risk

            # Overall validation
            is_valid = position_risk_valid and portfolio_risk_valid

            return {
                'is_valid': is_valid,
                'position_risk_valid': position_risk_valid,
                'portfolio_risk_valid': portfolio_risk_valid,
                'position_risk': risk_metrics.position_risk,
                'portfolio_risk': risk_metrics.portfolio_risk,
                'max_position_risk': self.max_position_risk,
                'max_portfolio_risk': self.max_portfolio_risk,
                'warnings': self._generate_risk_warnings(risk_metrics)
            }

        except Exception as e:
            self.logger.error(f"❌ Position risk validation failed: {e}")
            return {
                'is_valid': False,
                'error': str(e)
            }

    def _generate_risk_warnings(self, risk_metrics: RiskMetrics) -> list[str]:
        """Generate risk warnings based on metrics."""
        warnings = []

        if risk_metrics.position_risk > self.max_position_risk * 0.8:
            warnings.append(f"High position risk: {risk_metrics.position_risk:.4f}")

        if risk_metrics.portfolio_risk > self.max_portfolio_risk * 0.8:
            warnings.append(f"High portfolio risk: {risk_metrics.portfolio_risk:.4f}")

        if risk_metrics.volatility_risk > 0.01:
            warnings.append(f"High volatility risk: {risk_metrics.volatility_risk:.4f}")

        if risk_metrics.risk_reward_ratio > 0 and risk_metrics.risk_reward_ratio < 1.0:
            warnings.append(f"Low risk-reward ratio: {risk_metrics.risk_reward_ratio:.2f}")

        return warnings

    def get_risk_limits(self) -> Dict[str, float]:
        """Get current risk limits."""
        return {
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_position_risk': self.max_position_risk,
            'default_volatility': self.default_volatility
        }

    def update_risk_limits(self, new_limits: Dict[str, float]):
        """Update risk limits."""
        try:
            if 'max_portfolio_risk' in new_limits:
                self.max_portfolio_risk = new_limits['max_portfolio_risk']
            if 'max_position_risk' in new_limits:
                self.max_position_risk = new_limits['max_position_risk']
            if 'default_volatility' in new_limits:
                self.default_volatility = new_limits['default_volatility']

            self.logger.info("✅ Risk limits updated")

        except Exception as e:
            self.logger.error(f"❌ Failed to update risk limits: {e}")

    async def stop(self):
        """Stop risk calculator."""
        try:
            self.logger.info("🛑 Stopping Risk Calculator...")
            self.is_initialized = False
            self.logger.info("✅ Risk Calculator stopped successfully")

        except Exception as e:
            self.logger.error(f"❌ Error stopping Risk Calculator: {e}")

# Convenience function
async def setup_risk_calculator(config: TradingConfig) -> Optional[RiskCalculator]:
    """Setup and initialize risk calculator."""
    try:
        risk_calculator = RiskCalculator(config)
        success = await risk_calculator.initialize()
        if success:
            return risk_calculator
        return None
    except Exception as e:
        logger.error(f"❌ Failed to setup risk calculator: {e}")
        return None
